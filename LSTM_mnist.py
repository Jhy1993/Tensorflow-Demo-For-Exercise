# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 16:32:57 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
To classify images using a recurrent neural network, 
wo consider every iamge row as a sequence of pixels.
Because mnist is 28*28, 
so we handle 28 sequences of 28 step for every sample
Reference:
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
"""

# parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 100

# Neural Network parameters
n_input = 28
n_steps = 18
n_hidden = 128
n_classes = 10

# input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    # translate input shape:(batch_size, n_steps, n_input)
    # into 'n_step' tensors list shape (batch_size, n_input)
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # define lstm cell
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out'] + biases['out'])

pred = RNN(x, weights, biases)

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y  =mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print()
        step += 1
    print("optimization Finished~")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("test accuracy: ", 
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

