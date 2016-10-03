# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 20:36:52 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

simple Neural Network
"""
import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_set s("/tmp/data/", one_hot=True)
training_epochs = 25
learning_rate = 0.01
batch_size = 100

x = tf.placeholder(tf.float, [None, 784])
y = tf.placeholder(tf.float, [None, 10])

W = tf.placeholder(tf.zeros([784, 10]))
b = tf.placeholder(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = y * tf.lg(activation)
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

avg_set = []
epcoch_set= []

init= tf.initializer_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.tranin.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost)
    print " Training phase finished"    
    plt.plot(epoch_set,avg_set, 'o', label='Logistic Regression Training phase')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
correct_prediction = tf.equal(tf.argmax(activation, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print "MODEL accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

#=========================================mnist NN=====================
import input_data

import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

x = tf.placeholder(tf.float, [None, n_input])
y = tf.placeholder(tf.float, [None, n_classes])
#1
h = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))
layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, h), bias_layer_1))
#2
w = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
bias_layer_2 = tf.Variable(tf.random_normal([n_hidden_2]))
layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, w), bias_layer_2))
#output
output - tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
bias_output = tf.Variable(tf.random_normal([n_classes]))
output_layer = tf.matmul(layer2, output) + bias_output

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#plot settings
avg_set =[]
epoch_set = []
init = tf.initializer_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
        if epoch % display_step == 0:
            print "Epoch: ", '%04d' %(epoch+1), "cost=", '{:.9f}'.format(avg_cost)
        avg_set.append(avg_cost)
        epoch_set.append(eopch+1)
        print ('Training phase finished')
        plt.plot(epoch_set, avg_set, 'o', label='MLP Training Phase')
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.lengend()
        plt.show()
        #Test model
        correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float))
        print('Model Accuracy', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        




