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

sys.path.append("..")
sys.path.append("../..")
dfx=pd.read_csv("Data/xtrain.txt",header=None , sep=",")
convx= dfx.values


def compareResult(out,lbl):
    if(out[0]>out[1]):
        return lbl[0]==1
    else:
        return lbl[1]==1



if __name__ == "__main__":

    # VideoPath = '20180314_100215_I1.mp4'

    # modelpath="Model/output_graph.pb"
    # modelpath = 'graph/Drowsy.pb'
    # modelpath = 'graph/mnist_model_graph.pb'

    modelpath = 'keraGraph/Drowsy1.pb'

    predictor_path = "shape_predictor_68_face_landmarks.dat"
    faces_folder_path = "images/123.png"


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



            # densetensor = sess.run([dense], feed_dict={conv2d_1_inp: X})
            # densetensor=densetensor[0]
            # print(densetensor)
            # print(densetensor.shape,Y.shape)
            # sahi=0
            # for i in range(Y.shape[0]):
            #     if(densetensor[i],Y[i]):
            #         sahi+=1
            # print(sahi)




            # # for indx in range(0, X.shape[0]):
            # for indx in range(X.shape[0]):
            #     # print(indx,flist[indx])
            #     lbl = Y[indx]
            #     img=np.array([X[indx]])
            #
            #     print(img.shape)
            #     densetensor = sess.run([dense], feed_dict={conv2d_1_inp: img})
            #     print(densetensor)



