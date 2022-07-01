from keras.models import Sequential
from keras.layers import Dense, Dropout
import  pandas as pd,numpy as np
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




# dfx=pd.read_csv("Data/xtrain.txt",header=None , sep=",")
# convx= dfx.values
# X = np.array(convx, dtype=np.float)
# print(X.shape)
#
# dfx1=pd.read_csv("Data/xtrain1.txt",header=None , sep=",")
# convx1= dfx1.values
# X1 = np.array(convx1, dtype=np.float)
# #print ("X1= ")
# #print(X1)
#
#
# dfy=pd.read_csv("Data/ytrain.txt",header=None , sep=",")
# convy= dfy.values
# Y = np.array(convy, dtype=np.float)
# #print ("y= ")
# #print(Y)
# dfy1=pd.read_csv("Data/ytrain1.txt",header=None , sep=",")
# convy1= dfy1.values
# Y1 = np.array(convy1, dtype=np.float)
# #print ("Y1= ")
# #print(Y1)
#
# x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.20, random_state=42)
#
#
# min_max_scaler = StandardScaler()
# X_train_minmax = min_max_scaler.fit_transform(x_train)
#
#
#
#
# # X_test_minmax = min_max_scaler.transform(x_test)
# # X_test_minmax1= min_max_scaler.transform(X1)
#
# # X1 = np.expand_dims(x_train[0], 0)
# # print(X1)
# # X1 = min_max_scaler.fit_transform(X1)
# # #
# #
# # print(np.mean(X1),np.std(X1),np.max(X1),np.min(X1))
# # print(np.mean(X_train_minmax),np.std(X_train_minmax),np.max(X_train_minmax),np.min(X_train_minmax))
# # print(np.mean(X_test_minmax),np.std(X_test_minmax),np.max(X_test_minmax),np.min(X_test_minmax))
# # print(np.mean(X_test_minmax1),np.std(X_test_minmax1),np.max(X_test_minmax1),np.min(X_test_minmax1))
#
# data=[]
# mean=min_max_scaler.mean_
# var=min_max_scaler.var_
# data.append(mean)
# data.append(var)
#
# p=pd.DataFrame(data)
# p.to_csv("Data/meanvar.txt",header=None)
#
# # pd.read_csv("Data/xtrain.txt",header=None , sep=",")







# from sklearn.preprocessing import StandardScaler
# data = [[0,- 2], [0, 0], [1, 1], [1, 1]]
# scaler = StandardScaler()
# print(scaler.fit(data))
# StandardScaler(copy=True, with_mean=True, with_std=True)
# print(scaler.mean_)
# print(scaler.var_)
# d=[[.20, -25]]
# # print(scaler.transform(data))
# print(scaler.transform(d))
# print((d-scaler.mean_)/np.sqrt(scaler.var_))

