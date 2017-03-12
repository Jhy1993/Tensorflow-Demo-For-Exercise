# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:

INPUT:

OUTPUT:

REFERENCE:

"""
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', 100, 'num of epoch')
flags.DEFINE_integer('training_iters', 10000, 'how many time iter')
flags.DEFINE_string('optimizer', 'adam', 'how to optimizer')
flags.DEFINE_integer('batch_size', 128, 'batch ')
flags.DEFINE_string('checkpoint_dir', './checkpoint_dir', 'checkpoint_dir dictionary')
flags.DEFINE_integer('display', 1000, 'every ? iter show once')

learning_rate = FLAGS.learning_rate
epoch_number = FLAGS.epoch_number
batch_size = FLAGS.batch_size
display = FLAGS.display
training_iters = FLAGS.training_iters
FEATURE_SIZE = 28
LABEL_SIZE = 10

n_input = FEATURE_SIZE
n_steps = 28
n_hidden = 128
n_output = LABEL_SIZE

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_output])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_output]))
}

def RNN(x, weights, biases):
    #[batch, n_step, n_input]
    x = tf.transpose(x, [1, 0, 2])
    #[n_step, batch, n_input]
    x = tf.reshape(x, [-1, n_input])
    #[n_step*batch, n_input]
    x = tf.split(0, n_steps, x)
    #n_step 个 [batch, n_input]

    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

print('Use optimzier: {}'.format(FLAGS.optimizer))

if FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



saver = tf.train.Saver()
checkpoint_dir = FLAGS.checkpoint_dir
checkpoint_file = checkpoint_dir + '/checkpoint.ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % FLAGS.display == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print('Iter:{}, Loss:{}, Acc:{}'.format(step, loss, acc))
        step += 1
    print('Optimzie is Ok~')

    test_len = 128
    test_data =mnist.test.images[:test_len].reshap((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    test_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label})
    print('Test Acc: {}'.format(test_acc))

