import cv2
import numpy as np

# Initialize video capture from file
cap = cv2.VideoCapture('cam2.mkv')

# Read the first frame
_, first_frame = cap.read()

# Initialize background subtractors
fgb = cv2.bgsegm.createBackgroundSubtractorMOG()
BS_knn = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=300, detectShadows=True)
BS_MOG2 = cv2.createBackgroundSubtractorMOG2(varThreshold=20)

# Counter initialization
i = 0

def centre(x, y, w, h):
    """Calculate the center of a bounding box"""
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends
    
    frame_copy = frame.copy()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Apply background subtraction
    fgbmask = BS_MOG2.apply(gray)
    
    # Apply thresholding
    _, thresh_diff = cv2.threshold(blur, 118, 255, 0)
    
    # Apply Canny edge detection
    canny = cv2.Canny(gray, 50, 50)
    
    # Morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilate_canny = cv2.dilate(canny, kernel, iterations=5)
    
    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 1, 21, 5)
    
    # Find contours
    contours, _ = cv2.findContours(dilate_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process cropped region
    crop_gray = blur[350:480, 0:412]
    crop_frame = frame[350:480, 0:412]
    canny_crop = cv2.Canny(crop_gray, 50, 50)
    kernel_crop = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_canny_crop = cv2.dilate(canny_crop, kernel_crop, iterations=3)
    conts, _ = cv2.findContours(dilate_canny_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw a reference line
    cv2.line(crop_frame, (0, 30), (412, 30), (0, 255, 0), 2)
    
    if len(conts) != 0:
        big_contour = conts[0]  # Select the first contour
        x, y, w, h = cv2.boundingRect(big_contour)
        cv2.rectangle(crop_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = centre(x, y, w, h)
        
        if 25 < cy < 35:
            i += 1
            print(f"Object Count: {i}")
    
    # Display processed images
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Dilated Canny Crop', dilate_canny_crop)
    cv2.imshow('Adaptive Threshold', adaptive_thresh)
    
    # Exit condition
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
