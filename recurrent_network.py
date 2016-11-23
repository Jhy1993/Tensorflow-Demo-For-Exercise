# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:16:38 2016

@author: Jhy1993

https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.op import rnn, rnn_cell
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 #input_dim
n_steps = 28 #timesteps
n_hidden = 128
n_classes = 10

#
x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

#
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

#
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#
def RNN(x, weights, biases):
    # Prepare data shape
    # Current data input: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensor list of shape(batch_size, n_input)

    # Premuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_step*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensor of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops.html#split

    # Define a lstm cell
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get LSTM output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
