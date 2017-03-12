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

id_mat = tf.diag([11.0, 10., 1.2])
print(sess.run(id_mat))

A = tf.truncated_normal([2, 3])
print(sess.run(A))

print(sess.run(tf.matmul(A, id_mat)))

