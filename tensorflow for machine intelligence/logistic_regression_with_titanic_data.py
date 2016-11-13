# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 22:21:00 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:
 
Reference:
TensorFlow For Machine Intelligence P124
"""
import tensorflow as tf 

W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0.0, name="bias")

def combine_inputs(X):
    return tf.matmul(X, W) + b

def inference(X):
    return tf.sigmoid(combine_inputs(X))

def loss(X, Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(combine_inputs(X), Y))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/" + file_name])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)

