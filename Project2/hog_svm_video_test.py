"""
HOG + SVM Classifier for Image and Video Testing
-------------------------------------------------
This script loads a pre-trained SVM model (trained on HOG features)
and tests it on both:
1. **Image Dataset** - Predicts object category from test images.
2. **Live Video (or Pre-recorded Video)** - Detects objects in video frames.

Press **'q'** to exit the video stream.
"""

import cv2
import numpy as np
import os
import joblib
from skimage import feature, exposure

# Load pre-trained SVM model
MODEL_PATH = "hog_svm_model.pkl"  # Ensure the correct filename
model = joblib.load(MODEL_PATH)

# Define constants for HOG feature extraction
HOG_SIZE = (64, 32)  # Resized image size for HOG
PIXELS_PER_CELL = (10, 10)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

# ---------------------------
# 1️⃣ IMAGE CLASSIFICATION
# ---------------------------
def classify_test_images(test_image_path):
    """
    Classifies test images using HOG + SVM.

    Args:
    - test_image_path: Path to the test image folder.

    Returns:
    - Displays prediction results with HOG visualization.
    """
    if not os.path.exists(test_image_path):
        print(f"Error: Test image path '{test_image_path}' not found.")
        return

    test_images = os.listdir(test_image_path)
    print(f"Found {len(test_images)} test images.")

    for image_file in test_images:
        image_path = os.path.join(test_image_path, image_file)

        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Couldn't read image '{image_file}'. Skipping.")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, HOG_SIZE)

        # Extract HOG features
        hog_features, hog_image = feature.hog(
            resized_image, orientations=ORIENTATIONS,
            pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK,
            transform_sqrt=True, block_norm="L1", visualize=True
        )

        # Make prediction
        prediction = model.predict(hog_features.reshape(1, -1))[0]

        # Scale and display HOG visualization
        hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255)).astype("uint8")

        cv2.imshow(f"HOG Features - {image_file}", hog_image)
        cv2.putText(image, prediction.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(f"Test Image - {image_file}", image)

        cv2.waitKey(0)  # Wait for keypress before moving to next image

    cv2.destroyAllWindows()


# ---------------------------
# 2️⃣ VIDEO TESTING (Live or Pre-recorded)
# ---------------------------
def classify_video(video_path=None):
    """
    Performs object detection in video using HOG + SVM.

    Args:
    - video_path: Path to the video file (if None, uses webcam).

    Press 'q' to exit.
    """
    if video_path:
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    if not cap.isOpened():
        print("Error: Couldn't open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define ROI (Region of Interest)
        cropped_roi = gray_frame[230:480, 0:412]  # Adjust as needed

        # Apply threshold to filter out noise
        _, thresh = cv2.threshold(cropped_roi, 115, 255, cv2.THRESH_BINARY)
        black_pixel_count = np.sum(thresh == 0)

        # If sufficient dark pixels exist, proceed with classification
        if black_pixel_count > 0:
            resized_roi = cv2.resize(cropped_roi, HOG_SIZE)

            # Extract HOG features
            hog_features = feature.hog(
                resized_roi, orientations=ORIENTATIONS,
                pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK,
                transform_sqrt=True, block_norm="L1"
            )

            # Predict using SVM model
            prediction = model.predict(hog_features.reshape(1, -1))[0]
            confidence_score = model.decision_function(hog_features.reshape(1, -1))

            print(f"Prediction: {prediction} | Confidence: {confidence_score}")

            # Display prediction on video frame
            cv2.putText(frame, prediction.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Display video and processed frames
        cv2.imshow("Live Video", frame)
        cv2.imshow("Processed ROI", thresh)

        # Press 'q' to quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# Run the Script
# ---------------------------
if __name__ == "__main__":
    TEST_IMAGE_PATH = "TEST_IMAGE"  # Change this to the actual test image folder
    VIDEO_PATH = "cam2.mkv"  # Change to None for webcam

    print("\nStarting image classification...")
    classify_test_images(TEST_IMAGE_PATH)

    print("\nStarting video classification...")
    classify_video(VIDEO_PATH)
