import cv2
import numpy as np
import face_recognition

imgVirat = face_recognition.load_image_file('Images/Virat_Kohli.jpg')
imgVirat = cv2.cvtColor(imgVirat,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/Virat_Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgVirat)[0]
encodeVirat = face_recognition.face_encodings(imgVirat)[0]
cv2.rectangle(imgVirat,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(240,10,50),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(240,10,50),2)

results = face_recognition.compare_faces([encodeVirat],encodeTest)
faceDis = face_recognition.face_distance([encodeVirat],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Virat Kohli',imgVirat)
cv2.imshow('Virat Test',imgTest)
cv2.waitKey(0)