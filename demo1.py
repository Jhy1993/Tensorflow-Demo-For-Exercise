#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf 
import numpy as np 
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

W = tf.Variable(tf.random_uniform([1, 2], -1, 1))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(W, x_data) + b 
#最小化
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#初始化
init = tf.initialize_all_variables()
#启动计算图
sess = tf.Session()
sess.run(init)
#拟合
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sees.run(W), sess.run(b)





