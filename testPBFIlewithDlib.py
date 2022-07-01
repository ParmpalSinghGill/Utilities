import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import cv2
import math
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from io import StringIO
from PIL import Image
import dlib

# This is needed since the code is stored in the object_detection folder.
from numpy import float32



dfx=pd.read_csv("Data/xtrain.txt",header=None , sep=",")
convx= dfx.values
mean=convx[0]
sd=np.sqrt(convx[1])
min_max_scaler = StandardScaler()
min_max_scaler.mean_=convx[0]
min_max_scaler.var_=convx[1]

def compareResult(out,lbl):
    if(out[0]>out[1]):
        return lbl[0]==1
    else:
        return lbl[1]==1



if __name__ == "__main__":

    # VideoPath = '20180314_100215_I1.mp4'

    # modelpath="Model/output_graph.pb"
    modelpath = 'graph/Drowsy.pb'
    # modelpath = 'graph/mnist_model_graph.pb'

    # modelpath = 'keraGraph/Drowsy.pb'

    predictor_path = "shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # modelpath = 'tensorflow_model/Graph1.pb'

    datapath="Data/xtrain.txt"
    lblpath="Data/ytrain.txt"
    dfx = pd.read_csv(datapath, header=None, sep=",")
    dfy = pd.read_csv(lblpath, header=None, sep=",")
    convx = dfx.values
    X = np.array(convx, dtype=np.float)
    convy = dfy.values
    Y = np.array(convy, dtype=np.float)
    min_max_scaler = StandardScaler()
    X = min_max_scaler.fit_transform(X)



    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(modelpath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    count = 0

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # get list of all tensor
            for f in detection_graph.get_operations():
                print(f.name,'  ',f.values())
            # Definite input and output Tensors for detection_graph
            conv2d_1_inp=detection_graph.get_tensor_by_name("dense_1_input:0")
            dense=detection_graph.get_tensor_by_name("dense_4/Softmax:0")

            cap = cv2.VideoCapture(0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            while (cap.isOpened()):
                ret, img = cap.read()
                if ret == False:
                    break

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

                    densetensor = sess.run([dense], feed_dict={conv2d_1_inp: mscaler})
                    res = densetensor[0][0]
                    if res[0] > res[1]:
                        text = " 0"
                    else:
                        text = " 1"

                    print(text)
                    cv2.putText(img, text, (10, 350), font, 1, (255, 5, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, str(k)+str(res), (10, 450), font, 1, (255, 5, 255), 1, cv2.LINE_AA)

                cv2.imshow('MAN', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


            # densetensor = sess.run([dense], feed_dict={conv2d_1_inp: X})
            # densetensor=densetensor[0]
            # print(densetensor)
            # print(densetensor.shape,Y.shape)
            # sahi=0
            # for i in range(Y.shape[0]):
            #     if(densetensor[i],Y[i]):
            #         sahi+=1
            # print(sahi)




