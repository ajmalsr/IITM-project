import cv2
import numpy as np
cap=cv2.VideoCapture('path of the video')
i =0
fps = cap.get(5)
psg = 0
psy = 0
# redd = []
# timer = []
# prestate = 0
while (True):
    # ff=cap.get(1)
    # print(ff)
    # print(i)
    ret, frame = cap.read()
    green = frame[119:135, 743:757]
    yellow = frame[100:119, 743:757]
    red = frame[87:101, 743:757]
    redbig = cv2.resize(red, (60, 60))
    greenbig = cv2.resize(green, (60, 60))
    yellowbig = cv2.resize(yellow, (60, 60))
    green_gray = cv2.cvtColor(greenbig, cv2.COLOR_BGR2GRAY)
    red_gray = cv2.cvtColor(redbig, cv2.COLOR_BGR2GRAY)
    yellow_gray = cv2.cvtColor(yellowbig, cv2.COLOR_BGR2GRAY)
    _, threshg = cv2.threshold(green_gray, 100, 255, 1)
    _, threshr = cv2.threshold(red_gray, 100, 255, 1)
    _, threshy = cv2.threshold(yellow_gray, 100, 255, 1)
    g = np.sum(threshg == 0)
    r = np.sum(threshr == 0)
    y = np.sum(threshy == 0)
    # print(y)
    # redd.append(r)
    if g > 900:
        csg = 1
    else:
        csg = 0

    if csg != psg and csg == 1:
        print('green is ON at:' + str(i / fps) + 'sec')

    # yellow
    if y > 1000:
        csy = 1
    else:
        csy = 0
    if csy != psy and csy == 0:
        print('red is ON at:' + str(i / fps) + 'sec')
    if csy != psy and csy == 1:
        print('yellow is ON at:' + str(i / fps) + 'sec')
    if i==0 & csy==0 & csg==0:
        print('red is ON at:0sec')

    psg = csg
    psy = csy

    i += 1

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break