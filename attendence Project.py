import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

path='attendence_images'
images=[]
classNames=[]

mylist=os.listdir(path)

for cl  in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncoding(images):
    encodedImg=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedImg.append(encode)
    return encodedImg

def markAttendence(name):
    with open('attendence.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry)
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


encodedListKnown=findEncoding(images)

print('encoding complete')

cap=cv2.VideoCapture(0)


while True:
    success,img=cap.read()
    imgSmall=cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall,faceCurFrame)

    for encodeFce,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodedListKnown,encodeFce)
        distance=face_recognition.face_distance(encodedListKnown,encodeFce)
        matchIndex=np.argmin(distance)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(img,name,(x1,y2),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)
            markAttendence(name)
        else:
            cv2.putText(img,"UNKNOWN FACE", (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            markAttendence("UNKNOWN PERSON")


    cv2.imshow("webcame",img)
    cv2.waitKey(1)



