# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Created on Tue Dec 13 13:41:25 2016

@author: Jhy
"""
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected
=======
Created on Thu Dec  8 18:44:18 2016

@author: Jhy1993
"""

import numpy as np
import tensorflow as tf

graph = tf.Graph()
m1 = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=np.float32)

with graph.as_default():
    m1_input = tf.placeholder(tf.float32, shape=[4, 2])
    m2 = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))
    m3 = tf.matmul(m1_input, m2)
    m3 = tf.Print(m3, [m3], message='m3 is: ')
    m4 = tf.initialize_all_vairables()
    
with tf.Session(graph=graph) as sess:
    int
    
    
>>>>>>> fe2084bae4e3a57e13a08050bb826a368bc64f1c
