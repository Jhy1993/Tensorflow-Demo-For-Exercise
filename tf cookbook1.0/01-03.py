# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:48:05 2017

@author: jhy
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()
rand_array = np.random.rand(4, 4)

x = tf.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x)

print(sess.run(y, feed_dict={x: rand_array}))


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/va0103', sess.graph)
