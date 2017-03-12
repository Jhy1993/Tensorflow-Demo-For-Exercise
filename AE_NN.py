# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:20:02 2016

@author: Jhy_BISTU
README：
基于自编码器/神经网络的WiFi指纹地点识别】
’Place recognition with WiFi fingerprints using Autoencoders and Neural Networks'
INPUT:

OUTPUT:

REFERENCE:
https://arxiv.org/abs/1611.02049
https://github.com/aqibsaeed/Place-Recognition-using-Autoencoders-and-NN/blob/master/Place%20recognition%20with%20WiFi%20fingerprints%20using%20AE%20and%20NN.ipynb
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale


dataset = pd.read_csv("trainingData.csv",header = 0)
features = scale(np.asarray(dataset.ix[:,0:520]))
labels = np.asarray(dataset["BUILDINGID"].map(str) + dataset["FLOOR"].map(str))
labels = np.asarray(pd.get_dummies(labels))

train_val_split = np.random.rand(len(features)) < 0.70
train_x = features[train_val_split]
train_y = labels[train_val_split]
val_x = features[~train_val_split]
val_y = labels[~train_val_split]

test_dataset = pd.read_csv("validationData.csv",header = 0)
test_features = scale(np.asarray(test_dataset.ix[:,0:520]))
test_labels = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))
test_labels = np.asarray(pd.get_dummies(test_labels))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


n_input = 520 
n_hidden_1 = 256 
n_hidden_2 = 128 
n_hidden_3 = 64 

n_classes = labels.shape[1]

learning_rate = 0.01
training_epochs = 20
batch_size = 10

total_batches = dataset.shape[0] // batch_size

X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.float32, shape=[None, n_classes])
# Encoder
e_weights_h1 = weight_variable([n_input, n_hidden_1])
e_biases_h1 = bias_variable([n_hidden_1])

e_weights_h2 = weight_variable([n_hidden_1, n_hidden_2])
e_biases_h2 = weight_variable([n_hidden_2])

e_weights_h3 = weight_variable([n_hidden_2, n_hidden_3])
e_biases_h3 = bias_variable([n_hidden_3])

# Decoder
d_weights_h1 = weight_variable([n_hidden_3, n_hidden_2])
d_biases_h1 = bias_variable([n_hidden_2])

d_weights_h2 = weight_variable([n_hidden_2, n_hidden_1])
d_biases_h2 = bias_variable([n_hidden_1])

d_weights_h3 = weight_variable([n_hidden_1, n_input])
d_biases_h3 = bias_variable([n_input])

# DNN
dnn_weights_h1 = weight_variable([n_hidden_3, n_hidden_2])
dnn_biases_h1 = bias_variable([n_hidden_2])

dnn_weights_h2 = weight_variable([n_hidden_2, n_hidden_2])
dnn_biases_h2 = bias_variable([n_hidden_2])

dnn_weights_out = weight_variable([n_hidden_2, n_classes])
dnn_biases_out = bias_variable([n_classes])

def encoder(x):
    L1 = tf.nn.tanh(tf.add(tf.matmul(x, e_weights_h1), e_biases_h1))
    L2 = tf.nn.tanh(tf.add(tf.matmul(L1, e_weights_h2), e_biases_h2))
    L3 = tf.nn.tanh(tf.add(tf.matmul(L2, e_weights_h3), e_biases_h3))
    return L3

def decoder(x):
    L1 = tf.nn.tanh(tf.add(tf.matmul(x, d_weights_h1), d_biases_h1))
    L2 = tf.nn.tanh(tf.add(tf.matmul(L1, d_weights_h2), d_biases_h2))
    L3 = tf.nn.tanh(tf.add(tf.matmul(L2, d_weights_h3), d_biases_h3))
    return L3

def dnn(x):
    L1 = tf.nn.tanh(tf.add(tf.matmul(x, dnn_weights_h1), dnn_biases_h1))
    L2 = tf.nn.tanh(tf.add(tf.matmul(L1, dnn_weights_h2), dnn_biases_h2))
    L3 = tf.nn.tanh(tf.add(tf.matmul(L2, dnn_weights_h2), dnn_biases_out))
    return L3


encoded = encoder(x)
decoded = decoder(encoded)
y_ = dnn(encoded)

us_cost = tf.reduce_mean(tf.pow(X - decoded, 2))
s_cost = tf.reduce_mean(tf.pow(Y * tf.log(y_)))

us_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(us_cost)
s_opt = tf.tranin.GradientDescentOptimizer(learning_rate).minimize(s_cost)

correct_predict = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    # Train AE ---Unsupervised Learning
    for epoch in range(training_epochs):
        epoch_costs = np.empty(0)
        for b in range(total_batches):
            offset = (b * batch_size) % (features.shape[0] - batch_size)
            batch_x = features([offset])

