# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:20:02 2016

@author: Jhy_BISTU
README：
VAE: Variational Autoencoder in Tensorflow
INPUT:

OUTPUT:

REFERENCE:
https://github.com/vaxin/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/variational_autoencoder.py
变分自编码器（Variational Autoencoder, VAE）通俗教程 – 邓范鑫——致力于变革未来的智能技术
"""
from __feature__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class Layer:
    """docstring for Layer"""
    def __init__(self, input, n_output):        
        self.input = input
        W = tf.Variable(tf.truncated_normal([int(self.input.get_shape()[1]), n_output], stddev=0.001))
        b = tf.Variable(tf.constant(0.0, shape=[n_output]))
        self.raw_output = tf.matmul(input, W) + b
        self.n_output = tf.nn.relu(self.raw_output)

# Dim of data
n_x = 784
# Dim of latent variabls count
n_z = 20
X = tf.placeholder(tf.float32, shape=[None, n_x])

# \mu(x) 2-layer
ENCODER_HIDDEN_COUNT = 400
mu = Layer(Layer(X, ENCODER_HIDDEN_COUNT).output, n_z).raw_output

# \Sigma(x) 2-layer
log_sigma = Layer(Layer(x, ENCODER_HIDDEN_COUNT).output, n_z).raw_output
sigma = tf.exp(log_sigma)

# KLD