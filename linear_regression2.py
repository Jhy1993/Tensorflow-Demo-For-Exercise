# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 20:36:52 2016

@author: Jhy1993
Github: https://github.com/Jhy1993


"""
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
number_of_points = 500
x_point = []
y_point = []
a = 0.22
b = 0.78
for i in range(number_of_points):
    x = np.ran dom.normal(0.0,0.5)
    y = a*x + b +np.random.normal(0.0,0.1)
    x_point.append([x])
    y_point.append([y])


plt.plot(x_point,y_point, 'o', label='Input Data')
plt.legend()
plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = W* x_point + b
cost = tf.reduce_mean(tf.square(y - y_point))
optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(cost)
model = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(model)
    for step in range(0, 21):
        sess.run(train)
        if (step % 5) == 0:
            plt.plot(x_point, y_point, 'o', label='step={}'.format(step))
            plt.plot(x_point, sess.run(W) * x_point + sess.run(b))
            plt.legend()
            plt.show()
            