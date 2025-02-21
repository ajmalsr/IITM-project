import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure, feature

# Load and preprocess image
image_path = "Frame85.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_img = resize(image, (128, 64))

# Apply Canny edge detection
canny = cv2.Canny(gray, 50, 50)

# Apply thresholding
_, thresh = cv2.threshold(gray, 118, 255, 0)

# Compute HOG features
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
(H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), transform_sqrt=True, 
                            block_norm="L1", visualize=True)

# Rescale HOG images for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255)).astype("uint8")

# Display images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
ax1.imshow(resized_img, cmap=plt.cm.gray)
ax1.set_title('Resized Input Image')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

# Display processed images using OpenCV
cv2.imshow("HOG Image", hogImage)
cv2.imshow("Thresholded Image", thresh)
cv2.imshow("Canny Edge Detection", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print shape information
print(f"HOG Feature Vector Shape: {H.shape}")
print(f"Original Image Shape: {image.shape}")
