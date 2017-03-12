# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:51:22 2017

@author: jhy
"""
import numpy as np 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
max_sample = 400000
batch_size = 128
display_step = 10
n_input = 28
n_steps = 28
n_hidden = 256
n_classes = 10

weights = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

