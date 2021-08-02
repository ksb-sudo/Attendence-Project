import cv2
import numpy as np
import face_recognition

imgElon=face_recognition.load_image_file('images/elon musk.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgtest=face_recognition.load_image_file('images/elon test.jpg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
top,right,bottom,left=faceLoc[0],faceLoc[1],faceLoc[2],faceLoc[3]
cv2.rectangle(imgElon,(left,top),(right,bottom),(0,255,0),1)

faceLocTest=face_recognition.face_locations(imgtest)[0]
encodeElonTest=face_recognition.face_encodings(imgtest)[0]
top,right,bottom,left=faceLocTest[0],faceLocTest[1],faceLocTest[2],faceLocTest[3]
cv2.rectangle(imgtest,(left,top),(right,bottom),(255,255,0),1)

results=face_recognition.compare_faces([encodeElon],encodeElonTest)
distance=face_recognition.face_distance([encodeElon],encodeElonTest)

cv2.putText(imgtest,f'{results} {round(distance[0],2)}',(50,50),cv2.FONT_ITALIC,2,(255,0,200),2)

print(results,distance)


#cv2.imshow("ELON MUSK",imgElon)
cv2.imshow("ELON TEST",imgtest)

cv2.waitKey(0)

