# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:
MLP with xavier
INPUT:

OUTPUT:

REFERENCE:
https://github.com/sjchoi86/Tensorflow-101/blob/master/notebooks/mlp_mnist_xavier.ipynb
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

def xavier_init(n_input, n_output, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_input + n_output))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_input, n_output))
        return tf.truncated_normal_initializer(stddev=stddev)
        
lr = 0.01
train_epoch = 50
batch_size = 100
display_step = 1

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

x = tf.placeholder('float', [None, n_input])     
y = tf.placeholder('float', [None, n_classes])
keep_prob = tf.placeholder('float')

        
        
        
        
        
        
        
    