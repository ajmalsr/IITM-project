"""
HOG + SVM Vehicle Detector with Motion Tracking
------------------------------------------------
This script detects moving vehicles in a video stream using:
1. **HOG (Histogram of Oriented Gradients) features** for object representation.
2. **SVM (Support Vector Machine) classifier** for vehicle classification.
3. **Sliding window & image pyramids** for multi-scale detection.
4. **Non-Maximum Suppression (NMS)** for refined bounding boxes.
5. **Motion Tracking using Background Subtraction (MOG2)** to ignore static objects.

Press **'q'** to exit the video stream.
"""

import cv2
import numpy as np
import joblib
from skimage import feature
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from testingsvm2 import sliding_window  # Ensure this module is available

# Load the pre-trained SVM model
MODEL_PATH = "hog_svm_model.pkl"
model = joblib.load(MODEL_PATH)

# Define HOG parameters
HOG_SIZE = (64, 32)
PIXELS_PER_CELL = (10, 10)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

# Define sliding window parameters
WINDOW_SIZE = (80, 120)  # (width, height)
DOWNSCALE = 1.5  # Image pyramid scaling factor
STEP_SIZE = 15  # Sliding window step size

# Initialize Background Subtractor for Motion Tracking
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)


def process_video(video_path):
    """
    Processes a video frame-by-frame for vehicle detection with motion tracking.

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

        # Apply motion tracking (Background Subtraction)
        fg_mask = bg_subtractor.apply(gray_frame)

        # Define Region of Interest (ROI)
        cropped_roi = gray_frame[230:480, 0:412]
        color_roi = frame[230:480, 0:412]  # Copy for displaying bounding boxes

        # Apply motion tracking to the ROI
        motion_roi = fg_mask[230:480, 0:412]
        motion_detected = np.sum(motion_roi == 255) > 500  # Threshold for motion presence

        # If motion is detected, proceed with object detection
        if motion_detected:
            detections = []

            # Loop over image pyramid layers
            for resized_image in pyramid_gaussian(cropped_roi, downscale=DOWNSCALE):
                # Loop over sliding window positions
                for (x, y, window) in sliding_window(resized_image, stepSize=STEP_SIZE, windowSize=WINDOW_SIZE):
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
                    hog_features = hog_features.reshape(1, -1)

                    # Predict using SVM model
                    prediction = model.predict(hog_features)[0]
                    confidence_score = model.decision_function(hog_features)

                    if prediction == 1 and confidence_score > 0.6:  # 1 means 'vehicle'
                        print(f"Detection: ({x}, {y}) | Confidence: {confidence_score}")
                        detections.append((x, y, confidence_score, WINDOW_SIZE[0], WINDOW_SIZE[1]))

            # Apply Non-Maximum Suppression (NMS) to refine bounding boxes
            rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
            scores = np.array([score for (_, _, score, _, _) in detections])

            if len(rects) > 0:
                final_detections = non_max_suppression(rects, probs=scores, overlapThresh=0.1)

                # Draw final detections after NMS
                for (xA, yA, xB, yB) in final_detections:
                    cv2.rectangle(color_roi, (xA, yA), (xB, yB), (0, 255, 0), 2)

            # Display processed frames
            cv2.imshow("Detections", color_roi)
            cv2.imshow("Motion Mask", motion_roi)

        # Display original frame
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

    print("\nStarting video detection with motion tracking...")
    process_video(VIDEO_PATH)
