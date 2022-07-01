import os
import dlib
import glob
import cv2
import sys

predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = "images/123.png"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# import os, sys

count=0
cap = cv2.VideoCapture(0)

# directories = [ x for x in os.listdir('.') if os.path.isdir(x) ]
img=0
while(cap.isOpened()):
    ret, img = cap.read()
    #img = cv2.resize(images, (360, 360))
    #path = "D:\\"
    #print(fullPath)
    #cv2.imwrite(fullPath,img)

    cv2.imshow('Car detector', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

cv2.imwrite("photo4.png",img)

dets = detector(img, 1)
print("Length ",len(dets))
for k, d in enumerate(dets):
    shape = predictor(img, d)
    a = []
    for i in range(0, 68):
        x = shape.part(i).x
        y = shape.part(i).y
        a.append(x)
        a.append(y)
        # img = cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        img = cv2.putText(img, "*", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

# print("Number of faces detected: {}".format(len(dets)))
# if(int(format(len(dets)))>1):


cv2.imshow('68', img)
cv2.waitKey(10000)
