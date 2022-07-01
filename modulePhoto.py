import os
import dlib
import glob
import cv2
import sys

predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = "photo.png"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# import os, sys

count=0
cap = cv2.VideoCapture(0)
path="images"

for x1 in os.listdir(path):
	faces_folder_path=path+"/"+x1
	img = cv2.imread(faces_folder_path)
	# img = cv2.resize(img, (360, 360))

	dets = detector(img, 1)
	print("length is",len(dets))
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
	cv2.imshow(str(x1), img)
	cv2.waitKey(3000)
