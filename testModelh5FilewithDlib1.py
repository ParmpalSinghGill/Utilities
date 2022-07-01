from builtins import print
from sklearn.preprocessing import StandardScaler
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import dlib
from sklearn.model_selection import train_test_split
# from PIL import Image

datapath="Data/xtrain.txt"
lblpath="Data/ytrain.txt"
photo="images/photo.png"
predictor_path = "shape_predictor_68_face_landmarks.dat"
photo = "images/photo.png"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


model_path = 'Model/model1000-18.h5'
model_weights_path = 'Model/model_weights-18.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)




def compareResult(lbl,out):
    if(out[0]>out[1]):
        return lbl[0]==1
    else:
        return lbl[1]==1

import pandas as pd


dfx = pd.read_csv(datapath, header=None, sep=",")
dfy = pd.read_csv(lblpath, header=None, sep=",")
convx = dfx.values
X = np.array(convx, dtype=np.float)
convy = dfy.values
Y = np.array(convy, dtype=np.float)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
min_max_scaler = StandardScaler()
X_train_minmax = min_max_scaler.fit_transform(x_train)



font = cv2.FONT_HERSHEY_SIMPLEX
img=cv2.imread(photo)

dets = detector(img, 1)
print("Length ", len(dets))
for k, d in enumerate(dets):
    shape = predictor(img, d)
    a = []
    for i in range(0, 68):
        x = shape.part(i).x
        y = shape.part(i).y
        a.append(x)
        a.append(y)
    a=np.expand_dims(a,0)
    mscaler=min_max_scaler.transform(a)
    # print(mscaler)

    densetensor = model.predict(mscaler)
    res = densetensor[0]
    if res[0] > res[1]:
        text = " 0"
    else:
        text = " 1"

    print("imageSize ", img.shape)
    print("points")
    for i, p in enumerate(a[0]):
        print(i, p)
    print("SC -->")
    for i, p in enumerate(mscaler[0]):
        print(i, p)
    print("out ", densetensor)

    print(text)
    cv2.putText(img, text, (10, 350), font, 1, (255, 5, 255), 1, cv2.LINE_AA)
    cv2.putText(img, str(k)+str(res), (10, 450), font, 1, (255, 5, 255), 1, cv2.LINE_AA)

cv2.imshow('MAN', img)
cv2.waitKey(3000)
