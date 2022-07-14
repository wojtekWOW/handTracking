import cv2
import os
import numpy as np
import HandTrackingModule as htm

# object for prerecorded video
video = "Resources/gestures.mp4"
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")
#size of webcam window
h,w =720,1280
cap.set(3, w)
cap.set(4, h)

#path to images
folderPath = "Resources/gestures"
# list of images in Resources/gestures directory
myList = os.listdir(folderPath)
#list to store images in cv2 readable format
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

#object created from HandTrackingModule
detector = htm.handDetector()

#list of finger tips locators, 4 is thumb, 20 is pinky finger
tipIds = [4, 8, 12, 16, 20]

#variables for painting
xp, yp = 0, 0
drawColor = (0, 0, 230)
eraseColor = (0,0,0)
brushThickness = 20
eraseThickness = 100

#canvas for drawing on, later is merged with orginal image
imgCanvas = np.zeros((h, w,3), np.uint8)

#main program updating video frames
while True:
    success, img = cap.read()

    #flip video to remove mirror efect
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    #landmark list
    lmList = detector.findPosition(img)
    # if there are some landmarks detected
    if len(lmList) !=0:

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[16][1:]
        x4, y4 = lmList[20][1:]

        #list to store which fingers are raised
        fingers = []

        #thumb is raised
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 other fingers are raised
        for id in range(1,5):
            if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:  #tipIds[id][number 1 stand for x, number 2 stand for y position value
                fingers.append(1)
            else: fingers.append(0)

         # number of raised fingers
        imagenr = fingers.count(1)

        #gestures recognition other than number of fingers raised

        #thumbs up
        # check if all landmarks are lower than tip of a thumb
        thumbsup=True
        for id in range(0,20):
            if id==4: continue
            if lmList[4][2]>lmList[id][2]:
                thumbsup=False
        if thumbsup:
            imagenr=6

        # thumbs down
        # check if all landmarks are above tip of a thumb
        thumbsdown = True
        for id in range(0, 20):
            if id == 4: continue
            if lmList[4][2] < lmList[id][2]:
                thumbsdown = False
        if thumbsdown:
            imagenr=7

        # lets rock
        # check if only index and pinky finger are raised
        if fingers==[0,1,0,0,1]:
            imagenr = 8

        # ok
        # index finger is down and touching thumb
        if fingers == [1, 0, 1, 1, 1] and lmList[4][1]-lmList[8][1]<10 and lmList[4][2]-lmList[8][2]<10:
            imagenr = 9

        #painting
        #index and middle finger are up and touching
        if fingers == [0, 1, 1, 0, 0] and x2-x1<30:
            #draw a circle on index finger when ready to draw
            cv2.circle(img, (x1, y1),20,drawColor, cv2.FILLED)
            imagenr=11
            # if first frame start from position of a index finger
            if xp==0 and yp==0:
                xp, yp = x1, y1
            #draw lines from previous point to currenr point every frame
            cv2.line(img, (xp, yp),(x1,y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            #renew prevoius point as points from prevoius frame
            xp, yp = x1, y1
        #if not painting reset starting point
        else:
            xp, yp = x1, y1

        # erase
        # all fingers together
        if fingers == [0, 1, 1, 1, 1] and (x4 - x1) < 150 and (y4 - y1) < 50:
            imagenr=10
            # draw a circle on index finger when ready to draw
            cv2.circle(img, (x1, y1), 20, eraseColor, cv2.FILLED)

            #if first frame start from position of a index finger
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            # erase lines from previous point to currenr point every frame
            cv2.line(img, (xp, yp), (x1, y1), eraseColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), eraseColor, eraseThickness)
            xp, yp = x1, y1
        else:
            xp, yp = x1, y1

        img[0:200, 0:200] = overlayList[imagenr-1] #0 is last alement and index -1 gets last element in a list

    #change canvas to gray scale image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    #invert gray clolors so black background is 0
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    #revert to colors
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    #bitwise overlay image of a hand with canvas that something writen on
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    #opencv function to show display from camera
    cv2.imshow("Video", img)
    #cv2.imshow("Canvas", imgCanvas)

    # q exits program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break