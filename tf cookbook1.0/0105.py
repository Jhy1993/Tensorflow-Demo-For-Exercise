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

print(sess.run(tf.truediv(3, 4)))


seq = range(10)

def jhy(n):
    return (tf.sub(2 * tf.square(n), x) + 10)

for s in seq:
    print(sess.run(jhy(s)))