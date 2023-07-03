#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode=False,maxHand=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHand=maxHand
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHand,1,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

                 
    def findHands(self,img,Draw=True):

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)          ## hand tracking and recognition
        if self.results.multi_hand_landmarks:            ## (if available) Hands / Landmark detection (if hands are detected then this loop will exececute)
            for handLms in self.results.multi_hand_landmarks:
                if Draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    
                

    def findPosition(self,img,handNo=0,draw=True):
        #self.findHands(img)
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                h, w, c = img.shape                # height,weight and channells for image
                cx, cy = int(lm.x*w), int(lm.y*h)  # they are decimal places and converted into intergers (position of centered)
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)

        return lmList
    
def main():
    pTime=0
    cTime=0
    bre=0
    cap=cv2.VideoCapture(0)
    detector=handDetector()

    while True:
        success, img= cap.read()
        detector.findHands(img)
        lmList=detector.findPosition(img )
        
        if len(lmList)!=0:
            print(lmList[4])
        
        #print(f"detector output : ",detector.findHands(img))
        #print("this is  new image\n:",img)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
                   (255,255,255),3)
        
        cv2.imshow("Image",img)
        key=cv2.waitKey(1)
        
        if key==ord("q"):
            bre=1
            break
        if bre==1:
            cap.release()
            cv2.destroyAllWindows()

if __name__=="__main__":
    main()


# In[ ]:





# In[ ]:




