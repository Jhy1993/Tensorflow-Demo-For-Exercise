# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:
RNN classify mnist
timesteps = 28
feature = 28
INPUT:

OUTPUT:

REFERENCE:
https://www.youtube.com/watch?v=IASyrQamTQk&index=24&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 1000
batch_size = 128

n_inputs = 28
n_timesteps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
           'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
           'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
           }
biases = {
          'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,]))
          'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
          }

def RNN(X, weights, biases):
    # X(128 batch_size, 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul()
    lstm = tf.nn.rnn_cell.
      pass  
          
          
          
          
          
          
          
          
          
          












