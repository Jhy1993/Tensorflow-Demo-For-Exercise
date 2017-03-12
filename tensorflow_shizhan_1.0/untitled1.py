# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:08:24 2017

@author: jhy
"""
from keras.datasets import mnist
from keras.models import  Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.utils import np_utils
(x_train, y_train), (x_test, t_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshpe(x_test.shape[0], 28, 28, 1)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=[28, 28, 1]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
his = model.fit(x_train, y_train)




























