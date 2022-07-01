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


predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

meanvar="meanvar.txt"
model_path = 'Model/model1000-18.h5'
model_weights_path = 'Model/model_weights-18.h5'
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

import pandas as pd,math


mv = pd.read_csv(meanvar, header=None, sep=",")
mv=np.array(mv)
mean=mv[0,1:]
var=np.sqrt(mv[1,1:])
# print(mean)
# print(var)


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

        mscaler=(a-mean)/var
        # print(mscaler)
        print(mscaler)

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



