"""
HOG + SVM Vehicle Detector in Video
------------------------------------
This script detects vehicles in a video stream using:
1. **HOG (Histogram of Oriented Gradients) features** for object representation.
2. **SVM (Support Vector Machine) classifier** for vehicle classification.
3. **Sliding window & image pyramids** for multi-scale detection.
4. **Non-Maximum Suppression (NMS)** for refined bounding boxes.

Press **'q'** to exit the video stream.
"""

import cv2
import numpy as np
import joblib
from skimage import feature
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from hog_svm_vehicle_detector import sliding_window  

# Load the pre-trained SVM model
MODEL_PATH = "hog_svm_model.pkl"  # Ensure the correct filename
model = joblib.load(MODEL_PATH)

# Define HOG parameters
HOG_SIZE = (64, 32)  # Standard input size for HOG
PIXELS_PER_CELL = (10, 10)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

# Define sliding window parameters
WINDOW_SIZE = (80, 120)  # (width, height)
DOWNSCALE = 1.5  # Image pyramid scaling factor
STEP_SIZE = 15  # Sliding window step size


def process_video(video_path):
    """
    Processes a video frame-by-frame for vehicle detection.

    Args:
    - video_path: Path to the input video file.

    Press 'q' to exit.
    """
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Create a copy of the frame for visualization
        frame_copy = frame.copy()

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define Region of Interest (ROI)
        cropped_roi = gray_frame[230:480, 0:412]
        color_roi = frame[230:480, 0:412]  # Copy for displaying bounding boxes

        # Apply threshold to filter out noise
        _, thresh = cv2.threshold(cropped_roi, 115, 255, cv2.THRESH_BINARY)
        black_pixel_count = np.sum(thresh == 0)

        # If sufficient dark pixels exist, perform object detection
        if black_pixel_count > 0:
            detections = []

            # Loop over image pyramid layers
            for resized_image in pyramid_gaussian(cropped_roi, downscale=DOWNSCALE):
                # Loop over sliding window positions
                for (x, y, window) in sliding_window(resized_image, stepSize=STEP_SIZE, windowSize=WINDOW_SIZE):
                    # Ignore windows that are smaller than required size
                    if window.shape[0] != WINDOW_SIZE[1] or window.shape[1] != WINDOW_SIZE[0]:
                        continue

                    # Resize to match HOG model input size
                    window_resized = cv2.resize(window, HOG_SIZE)

                    # Extract HOG features
                    hog_features = feature.hog(
                        window_resized, orientations=ORIENTATIONS,
                        pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK,
                        transform_sqrt=True, block_norm="L1"
                    )
                    hog_features = hog_features.reshape(1, -1)  # Reshape for SVM input

                    # Predict using SVM model
                    prediction = model.predict(hog_features)[0]
                    confidence_score = model.decision_function(hog_features)

                    # Store detection if it meets the confidence threshold
                    if prediction == 1 and confidence_score > 0.6:  # 1 means 'vehicle'
                        print(f"Detection: ({x}, {y}) | Confidence: {confidence_score}")

                        # Save detection coordinates
                        detections.append((
                            int(x), int(y), confidence_score, int(WINDOW_SIZE[0]), int(WINDOW_SIZE[1])
                        ))

            # Apply Non-Maximum Suppression (NMS) to remove duplicate detections
            rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
            scores = np.array([score for (_, _, score, _, _) in detections])

            if len(rects) > 0:
                final_detections = non_max_suppression(rects, probs=scores, overlapThresh=0.1)

                # Draw final detections after NMS
                for (xA, yA, xB, yB) in final_detections:
                    cv2.rectangle(color_roi, (xA, yA), (xB, yB), (0, 255, 0), 2)

            # Display processed frames
            cv2.imshow("Detections", color_roi)
            cv2.imshow("Original Frame", frame_copy)

        # Press 'q' to quit
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# Run the Script
# ---------------------------
if __name__ == "__main__":
    VIDEO_PATH = "cam2.mkv"  # Change to None for webcam

    print("\nStarting video detection...")
    process_video(VIDEO_PATH)
