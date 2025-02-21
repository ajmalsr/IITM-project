import cv2
import numpy as np
cap=cv2.VideoCapture('D:\TRAFFIC\THERMAL\VID-1970-01-01_22-21-35.mkv')
while (True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (960, 720))
    cv2.imshow('ff',frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break