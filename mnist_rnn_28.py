# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:
RNN classify mnist
timesteps = 28
feature = 28
INPUT:

OUTPUT:

REFERENCE:
https://www.youtube.com/watch?v=IASyrQamTQk&index=24&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 1000
batch_size = 128


n_inputs = 28
n_timesteps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
           'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
           'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
           }
biases = {
          'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
          'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
          }

def RNN(X, weights, biases):
    # X(128 batch_size, 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])# ==> X(128*28, 28)
    X_in = tf.matmul(X, weights['in']) + biases['in']# ==> X(128*28, hidden128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units]) #==>X(128, 28, 128hidden)
    
    lstm_cell = tf.nn.rnn_cell.BasiLSTMCell(n_hidden_units, forget_bias=1.0,
                                            state_is_truple=True)
    #_init_state = lstm_cell.zero_state(batch_size dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in,
                                        initial_state=_init_state,
                                        time_major=False)
          
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)                      

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
          
init = tf.initialize_all_variables()          
          
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys,
                 istate: np.zeros((batch_size, 2*n_hidden_units))})

        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                           istate: np.zeros((batch_size, 2*n_hidden_units))})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                            istate: np.zeros((batch_size, 2*n_hidden_units))})
            print('Iter: {}, Minibatch Loss: {:.6f}, Train Acc: {:.5f}'.format(step*batch_size, loss, acc))
        step += 1
    print('Optimizer is OK')    
    test_len = 256
    test_data = mnist.test.images[:test_len].reshape[-1, n_steps, n_inputs]
    test_label = mnist.test.images[:test_len]
    print('Test ACC: ', sess.run(accuracy,
                                 feed_dict={x: test_data, y:test_label,
                                 istate: np.zeros((test_len, 2*n_hidden_units))}))


tf.cons






  

          
          
          












