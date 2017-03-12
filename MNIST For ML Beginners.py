# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:02:39 2017

@author: jhy

https://www.tensorflow.org/get_started/mnist/beginners
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.nn.softmax(tf.matmul(x, W) + b)



loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

show_step = 100
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i % show_step == 0:
        ac = sess.run(acc, feed_dict={x: batch_xs, y: batch_ys})
        los = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
        print('Epoch: {}, Acc: {:.2f}, Loss: {:.2f}'.format(i, ac, los))


print(sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
