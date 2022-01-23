# https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471
import cv2
import numpy as np
import face_recognition

# Microsoft Visual Studio 2022, dlib, cmake

imgDonald = face_recognition.load_image_file('imagesBasic/donald.jpg')
imgDonald = cv2.cvtColor(imgDonald, cv2.COLOR_BGR2RGB)

imgDonaldTest = face_recognition.load_image_file('imagesBasic/alec.jpg')
imgDonaldTest = cv2.cvtColor(imgDonaldTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgDonald)[0]
encodeDonald = face_recognition.face_encodings(imgDonald)[0]
cv2.rectangle(imgDonald, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2)
#print(faceLoc)

faceLocTest = face_recognition.face_locations(imgDonaldTest)[0]
encodeDonaldTest = face_recognition.face_encodings(imgDonaldTest)[0]
cv2.rectangle(imgDonaldTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255), 2)

results = face_recognition.compare_faces([encodeDonald], encodeDonaldTest)
faceDis = face_recognition.face_distance([encodeDonald], encodeDonaldTest)
print(results)
print(faceDis)

cv2.putText(imgDonaldTest, f'{results} {round(faceDis[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('donald', imgDonald)
cv2.imshow('donald_0', imgDonaldTest)
cv2.waitKey(0)