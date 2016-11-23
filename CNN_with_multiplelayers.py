# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 20:49:20 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

http://blog.csdn.net/u014595019/article/details/52728886
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

print(FLAGS.data_dir)

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_2d(x, W):
    return tf.nn.conv_2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    pass