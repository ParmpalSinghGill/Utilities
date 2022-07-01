from __future__ import print_function

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import time

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

img_rows, img_cols = 64, 64
start = time.time()

# time.sleep(10)  # or do something more productive

done = time.time()
elapsed = done - start
print(elapsed)



batch_size = 2000
num_classes = 2
epochs = 1000
np.random.seed(7)
dfx=pd.read_csv("Data/xtrain.txt",header=None , sep=",")
convx= dfx.values
X = np.array(convx, dtype=np.float)
print(X.shape)

dfx1=pd.read_csv("Data/xtrain1.txt",header=None , sep=",")
convx1= dfx1.values
X1 = np.array(convx1, dtype=np.float)
#print ("X1= ")
#print(X1)


dfy=pd.read_csv("Data/ytrain.txt",header=None , sep=",")
convy= dfy.values
Y = np.array(convy, dtype=np.float)
#print ("y= ")
#print(Y)
dfy1=pd.read_csv("Data/ytrain1.txt",header=None , sep=",")
convy1= dfy1.values
Y1 = np.array(convy1, dtype=np.float)
#print ("Y1= ")
#print(Y1)

x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.20, random_state=42)


# print("----------x_train---")
# print(x_train.shape)
# print("----------X_test---")
# print(x_test.shape)
#
#
# print("----------y_train.shape---")
# print(y_train.shape)
# print("----------y_test.shape---")
# print(y_test.shape)
# print("----------train ok---")
#
#
# print("----------X1.shape---")
# print(X1.shape)
# print("----------Y1.shape---")
# print(Y1.shape)
# print("----------train ok---")


#StandardScaler()
min_max_scaler = StandardScaler()
X_train_minmax = min_max_scaler.fit_transform(x_train)


X_test_minmax = min_max_scaler.transform(x_test)
X_test_minmax1= min_max_scaler.transform(X1)

# X_train_minmax=x_train/np.max(x_train)
# X_test_minmax=x_test/np.max(x_test)
# X_test_minmax1=X1/np.max(X1)


print(X_test_minmax[0])
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(136,)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.00001),
              metrics=['accuracy'])

history = model.fit(X_train_minmax, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=.2)

start = time.time()

score = model.evaluate(X_test_minmax, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

done = time.time()
elapsed = done - start
print("---------------------------time---")
print(elapsed)
print("----------------------------time---")

score1 = model.evaluate(X_test_minmax1, Y1, verbose=0)
print('Test loss other Test :', score1[0])
print('Test accuracy other Test:', score1[1])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save("Model/model1000-18.h5")
print("Saved model to disk")

from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("Model/model-18.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save_weights('Model/model_weights-18.h5')

# print(model.predict(X[0]))






