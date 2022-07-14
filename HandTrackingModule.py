import cv2
import mediapipe as mp #google libary for handtracking among other rhings
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplex = 1, detectionCon = 0.5, trackCon=0.5):
        self.mode = mode #image or video stream, true = static image, false = video stream
        self.maxHands = maxHands #maximum number of recognised hands
        self.modelComplex = modelComplex #complexity of a landmark model
        self.detectionCon = detectionCon #Minimum confidence value from the hand detection model for the detection to be considered successful.
        self.trackCon = trackCon #Minimum confidence value from the landmark-tracking model for the hand landmarks to be considered tracked successfully, or otherwise hand detection will be invoked automatically on the next input image.
        #initialize mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        #opencv uses blue green red for clolors, it has to be changed to rgb for other functions
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #mediapipe process image to recognise hands
        self.results = self.hands.process(imgRGB)

        #if hands are found on the image
        if self.results.multi_hand_landmarks:
            # there can be 2 hands in video so for each hand in the image
            for handLms in self.results.multi_hand_landmarks:
                # draw landmarks with connections between landmarks
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=False):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # attach x, y pixel  position to each landmark
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape # c stands for channels, there are 3 for 3 colors
                cx, cy = int(lm.x * w), int(lm.y * h) #landmark x and y positions are converted from percentage value to pixel value
                #landmark list contin id, x and y position of each detected landmark
                lmList.append([id, cx, cy])
        return lmList

#dummy code for example
def main():
    # object for prerecorded video
    cap = cv2.VideoCapture("Resources/gestures.mp4")

    detector = handDetector()
    # reading from video
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[4])

        # showing video
        cv2.imshow("Video", img)
        # press q to exit video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()