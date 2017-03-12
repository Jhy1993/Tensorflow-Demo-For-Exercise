# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:48:05 2017

@author: jhy
"""
import tensorflow as tf 
from tensorflow.python.framework import ops
ops.reset_default_graph()
a = tf.constant(1.0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('/tmp/vab', graph = sess.graph)


