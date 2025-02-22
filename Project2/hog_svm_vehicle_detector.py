"""
HOG + SVM Vehicle Detector
--------------------------
This script uses Histogram of Oriented Gradients (HOG) and a pre-trained SVM model 
to detect vehicles in an image using a sliding window approach with image pyramids.

Steps:
1. Load the trained HOG + SVM model.
2. Apply image pyramids to handle different object scales.
3. Slide a detection window over the image.
4. Extract HOG features and predict using the SVM model.
5. Use Non-Maximum Suppression (NMS) to refine detections.
"""

import cv2
import numpy as np
import os
import joblib
from skimage import color, feature
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression


def sliding_window(image, step_size, window_size):
    """
    Generator that yields sliding window patches from an image.
    
    Args:
    - image: Input image
    - step_size: Number of pixels to move the window per step
    - window_size: Tuple (width, height) defining the sliding window size
    
    Yields:
    - x, y: Top-left coordinates of the window
    - window: The image patch within the window
    """
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


if __name__ == "__main__":
    # Load pre-trained SVM model
    model = joblib.load("hog_svm_model.pkl")

    # Read the input image
    image_path = "Frame206.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Image '{image_path}' not found.")
        exit()

    print(f"Original Image Shape: {image.shape}")

    # Resize image for faster processing (optional)
    image = cv2.resize(image, (300, 200))

    # Define sliding window parameters
    (winW, winH) = (50, 100)  # Window size (should match training size)
    window_size = (winW, winH)
    downscale = 1.5  # Image pyramid downscale factor

    detections = []  # Stores detected bounding boxes
    scale = 0  # Scale tracking for resizing

    # Loop over image pyramid layers
    for resized_image in pyramid_gaussian(image, downscale=downscale):
        # Loop over sliding window positions
        for (x, y, window) in sliding_window(resized_image, step_size=10, window_size=(winW, winH)):
            # Ignore windows that are smaller than required size
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # Convert window to grayscale and resize to match model input
            gray_window = color.rgb2gray(window)
            gray_window = cv2.resize(gray_window, (64, 32))

            # Extract HOG features
            hog_features = feature.hog(
                gray_window, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1"
            )
            hog_features = hog_features.reshape(1, -1)  # Reshape for SVM input

            # Make prediction using the SVM model
            prediction = model.predict(hog_features)

            # Only consider detections with a confidence score above threshold
            confidence_score = model.decision_function(hog_features)
            if prediction == 1 and confidence_score > 0.6:  # 1 means 'vehicle'
                print(f"Detection: ({x}, {y}) | Scale: {scale} | Confidence: {confidence_score}")

                # Save detection coordinates
                detections.append((
                    int(x * (downscale ** scale)), int(y * (downscale ** scale)),
                    confidence_score, int(window_size[0] * (downscale ** scale)),
                    int(window_size[1] * (downscale ** scale))
                ))

        scale += 1  # Move to next scale level

    # Draw raw bounding boxes
    for (x, y, _, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    # Apply Non-Maximum Suppression (NMS) to remove duplicate detections
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    scores = np.array([score[0] for (_, _, score, _, _) in detections])

    print("Detection confidence scores:", scores)
    final_detections = non_max_suppression(rects, probs=scores, overlapThresh=0.3)

    # Draw final detections after NMS
    for (xA, yA, xB, yB) in final_detections:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Display final detections
    cv2.imshow("Detections After NMS", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
