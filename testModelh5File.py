from builtins import print
from sklearn.preprocessing import StandardScaler
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model


# from PIL import Image

datapath="Data/xtrain1.txt"
lblpath="Data/ytrain1.txt"


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

import pandas as pd

dfx=pd.read_csv(datapath,header=None , sep=",")
convx= dfx.values
X = np.array(convx, dtype=np.float)

dfy=pd.read_csv(lblpath,header=None , sep=",")
convy= dfy.values
Y = np.array(convy, dtype=np.float)
res=[]

min_max_scaler = StandardScaler()
X = min_max_scaler.fit_transform(X)

sahi=0
for indx in range(X.shape[0]):
    r=predict(np.array([X[indx]]))
    if(compareResult(Y[indx],r)):
        sahi+=1
print(sahi/X.shape[0])


#     res.append()
#     if(indx%1000==0):
#         print(indx,' ',predict(np.array([X[indx]])),Y[indx])
# res=np.array(res)
# y1=Y[:,0]
# y2=Y[:,1]
# print(len(y1[y1==0]),len(y2[y2==1]))
# print(len(y1[y1==1]),len(y2[y2==0]))
# r1=res[:,0]
# # print(res.shape)
# print(len(r1[r1==1]),len(r1[r1==0]))
#
#

