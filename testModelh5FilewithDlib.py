from builtins import print
from sklearn.preprocessing import StandardScaler
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
import dlib

# from PIL import Image

datapath="Data/xtrain.txt"
lblpath="Data/ytrain.txt"
photo="images/photo.png"
predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


model_path = 'Model/aa/model1000-18.h5'
model_weights_path = 'Model/aa/model_weights-18.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(x):
    array = model.predict(x)
    # print(array,Y)
    result = array[0]
    answer =(result[0])
    # print('answer ',answer)
    return result


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
# print(min_max_scaler.mean_)


cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
while (cap.isOpened()):
    ret, img = cap.read()
    if ret == False:
        break

    dets = detector(img, 1)
    # print("Length ", len(dets))
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
        if res[0] >= res[1]:
            text = "Class 0 with "+str(res[0])+" PROB"
        else:
            text = "Class 1 with "+str(res[1])+" PROB"
        # print(text)
        cv2.putText(img, text, (10, 450), font, 1, (255, 5, 255), 1, cv2.LINE_AA)
    if(len(dets)==0):
        cv2.putText(img, "No Face", (10, 450), font, 1, (255, 5, 255), 1, cv2.LINE_AA)

    cv2.imshow('MAN', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()