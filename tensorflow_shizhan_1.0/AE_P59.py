# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:24:54 2017

@author: jhy
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
b1 = tf.Variable(tf.zeros([300]))

W2 = tf.Variable(tf.zeros([300, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, 784])

keep_prob = tf.placeholder(tf.float32)

h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h1_drop = tf.nn.dropout(h1, keep_prob)
y = tf.nn.softmax(tf.matmul(h1_drop, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(0.2).minimize(cross_entropy)

tf.global_variables_initializer().run()

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

for i in range(300):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    if i % 100 == 0:
        train_acc = acc.eval({x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print('Epoch: {} acc: {:.2f}'.format(i, train_acc))


print(acc.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
