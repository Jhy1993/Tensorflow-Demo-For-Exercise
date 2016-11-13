# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:02:47 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:

Reference:
Getting Started with TensorFlow(2016.07).A4 (1)
Chapter 6 GPU Programming
"""

import tensorflow as tf

with tf.device('/cpu:0'):
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

c = tf.matmul(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print sess.run(c)