# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 20:51:05 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

Solving partial differential equations
"""

import tensorflow as tf 
import numpy as numpy
import matplotlib.pyplot as plt 

N = 500

u_init = np.zeros([N, N], dtype=np.float32)
for n in range(40):
    a, b = np.random.randint(0, N, 2)
    u_init[a, b] = np.random.uniform()
plt.imshow(U.eval())
plt.show()

ut_int = np.zeros([N, N], dtype=np.float32)

eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())
U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

U_ = U + eps * Ut 
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

step = tf.group(U.assign(U_), Ut.assign(Ut_))

tf.initialize_all_variables().run()
for i in range(100):
    step.run({eps: 0.03, damping: 0,04})
    if i % 50 ==0:
        clear_output()
        plt.imshow(U.eval())
        plt.show()
        