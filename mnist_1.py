# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 20:36:52 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

mnist By KNN
"""
import numpy as np 
import tensorflow as tf 
import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_pixels, train_list_values = mnist.train.next_batch(100)
test_pixels, test_list_values = mnist.train.next_batch(100)

train_pixel_tensor = tf.placeholder(tf.float, [Npne, 784])
test_pixel_tensor = tf.placeholder(tf.float, [784])

distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor, tf.neg(test_pixel_tensor))), reduction_indices=1)

pred = tf.arg_min(distance, 0)

accuracy = 0
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_list_values))
        nn_index = sess.run(pred,
            feed_dict={train_pixel_tensor: train_pixels, test_pixel_tensor: test_pixels[i,:]})
        print "Test N° ", i,"Predicted Class: ", np.argmax(train_list_values[nn_index]), "True Class: ", np.argmax(test_ list_of_values[i])
    if np.argmax(train_list_values[nn_index]) == np.argmax(test_list_of_values[i]):
        accuracy += 1./len(test_pixels)
print "Result = ", accuracy