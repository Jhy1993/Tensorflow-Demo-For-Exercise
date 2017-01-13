# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 22:21:00 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:
linear regression 
Reference:
TensorFlow For Machine Intelligence P124
"""
import tensorflow as tf 



W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0.0, name="bias")

def inference(X):
    return tf.matmul(X, W) + b

def loss(X, Y):
    Y_pred = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_pred))

def inputs():
    age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36],]
    blood = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395]
    return tf.to_float(age), tf._tofloat(blood)

def train(total_loss):
    learning_rate = 0.001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    print sess.run(inference([[80.0, 25.0]]))
    print sess.run(inference([[65.0, 25.0]]))
