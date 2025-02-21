import cv2
import numpy as np

# Load video file
cap = cv2.VideoCapture('cam2.mkv')

# Skip the first 82 frames to reach the desired frame
frame_count = 0
while True:
    ret, frame = cap.read()
    frame_count += 1
    if frame_count == 82:
        break

# Read the first frame and convert it to grayscale
_, first_frame = cap.read()
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Copy grayscale image for processing
gray1 = gray.copy()
gray2 = gray.copy()

# Apply histogram equalization for contrast enhancement
hist_equalized = cv2.equalizeHist(gray)

# Read the second frame and convert to grayscale
_, second_frame = cap.read()
second_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

# Apply Gamma Correction for brightness adjustment
gamma = 2
gamma_corrected = np.array(255 * (gray / 255) ** gamma, dtype='uint8')
cv2.imshow('Gamma Transformed', gamma_corrected)

# Define sharpening kernel
sharpening_kernel = np.array([[-0.33, -0.33, -0.33],
                              [-0.33,  3.66, -0.33],
                              [-0.33, -0.33, -0.33]])

# Apply sharpening filter on both frames
image_sharp1 = cv2.filter2D(src=gray2, ddepth=-2, kernel=sharpening_kernel)
image_sharp2 = cv2.filter2D(src=second_gray, ddepth=-2, kernel=sharpening_kernel)

# Perform Canny edge detection
canny1 = cv2.Canny(image_sharp1, 80, 100)
canny2 = cv2.Canny(image_sharp2, 80, 100)

# Bitwise AND operation on detected edges
bitwise_and_result = cv2.bitwise_and(canny1, canny2)

# Additional Canny edge detection on original grayscale frame
canny_original = cv2.Canny(gray1, 80, 100)

# Bitwise OR operation for combined edge detection
bitwise_or_result = cv2.bitwise_or(bitwise_and_result, canny_original)

# Display the processed images
cv2.imshow('Sharpened Edge Detection', canny2)
cv2.imshow('Combined Edge Detection', bitwise_or_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
