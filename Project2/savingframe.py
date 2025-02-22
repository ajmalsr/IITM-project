import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
cap=cv2.VideoCapture('D:\TRAFFIC\THERMAL\VID-1970-01-01_22-15-39.mkv')

i=0
while (True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (960, 720))
    if ret == False:
        break

    cv2.imwrite('thermal' + str(i) + '.jpg', frame)
    i+=1
cap.release()
cv2.destroyAllWindows()
# _,first_frame= cap.read()
# gray=cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
# cropp= first_frame[200:480,0:412]
# cv2.imshow('frame', cropp)
# cv2.waitKey(0)