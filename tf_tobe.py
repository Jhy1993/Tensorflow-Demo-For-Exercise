
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:16:38 2016

@author: Jhy1993

Reference:
http://www.52cs.org/?p=1157

"""

import tensorflow as tf 
import numpy as np 

hi_op = tf.constant('hi')
a = tf.constant(10)
b = tf.constant(32)
jia = tf.add(a, b)

with tf.Session() as sess:
    print (sess.run(hi_op))
    print (sess.run(jia))
#================SGD==============================
train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.33 + 10

X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - tf.mul(X, w) - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
show_step = 100
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    epoch = 1
    for i in range(10):
        for (x, y) in zip(train_x, train_y):
            _, w_value, b_value = sess.run([train_op, w, b], feed_dict={X: x, Y: y})
            if epoch % show_step == 0:
                print("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
            epoch += 1

#==============1. prepare data================




#=================2. receive  Hyper-parameters===========================
'''
TensorFlow底层使用了python-gflags项目，
然后封装成tf.app.flags接口，
使用起来非常简单和直观，
在实际项目中一般会提前定义命令行参数，
尤其在后面将会提到的Cloud Machine Learning服务中，
通过参数来简化Hyperparameter的调优
'''
#Define Hyper-parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', None, 'Number of Epoch to run')
flags.DEFINE_integer('batch_size', 1024, 'batch size in single GPU')
flags.DEFINE_integer('validation_batch_size', 1024, 'batch siez in single GPU')
flags.DEFINE_integer('"thread_number', 1 'Number of thread to read data')
# flags.DEFINE_string('model', 'wide_and_deep')
flags.DEFINE_string('optimizer', 'adagrad', 'optimizer to train')
flags.DEFINE_string("mode", "train", "option mode: train, train_from_scratch, inference")
#====================define NN model============
input_units = FEATURE_SIZE
hidden1_units = 10
hidden2_units = 10
hidden3_units = 10
hidden4_units = 10
output_units  = LABEL_SIZE

def full_connect(inputs, weights_shape, biases_shape):
    with tf.device('/cpu:0'):
        weights = tf.get_variable("weights", weights_shape, initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", biases_shape, initializer=tf.random_normal_initializer())
    return tf.matmul(inputs, weights) + biases

def full_connect_relu(inputs, weights_shape, biases_shape):
    return tf.nn.relu(full_connect(inputs, weights_shape, biases_shape))

def deep_inference(inputs):
    with tf.variable_scope("layer1"):
        layer = full_connect_relu(inputs, [input_units, hidden1_units], [hidden1_units])
    
    with tf.variable_scope("layer2"):
        layer = full_connect_relu(layer, [hidden1_units, hidden2_units], [hidden2_units])

    with tf.variable_scope("layer3"):
        layer = full_connect_relu(layer, [hidden2_units, hidden3_units], [hidden3_units])

    with tf.variable_scope("layer4"):
        layer = full_connect_relu(layer, [hidden3_units, hidden4_units], [hidden4_units])

    with tf.variable_scope("output"):
        layer = full_connect(layer, [hidden4_units, output_units], [output_units])
    return layer

print("Use the optimizer: {}".format(FLAGS.optimizer))
if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
elif FLAGS.optimizer == "momentum":
    optimizer = tf.train.MomentumOptimizer(learning_rate)
elif FLAGS.optimizer == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
elif FLAGS.optimizer == "rmsprop":
    optimizer = tf.tranin.RMSPropOptimizer(learning_rate)
else:
    print("Unkonw optimizer: {}, exit now".fotmat(FLAGS.optimizer))
    exit(1)
#======online learning===========================
with tf.Session() as sess:
    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(tensorboard_dir, sess.graph)
    sess.run(init_op)
    sess.run(tf.initializer_local_variables())

    if mode == "train" or mode == "train_from_scratch":
        if mode != "train_from_scratch":
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if cpkt and ckpt.model_checkpoint_path:
                print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
                saver.restore(sess.ckpt.model_checkpoint_path)











#========================define NN model==================
# Define the model
input_units = FEATURE_SIZE
hidden1_units = 10
hidden2_units = 10
hidden3_units = 10
hidden4_units = 10
output_units = LABEL_SIZE

def full_connect(inputs, weights_shape, biases_shape):
    with tf.device('/cpu:0'):
        weights = tf.get_variable("weights",
                                   weights_shape,
                                   initializer=tf.random_noraml_initializer())
        biases = tf.get_variable("biases",
                                  biases_shape,
                                  initializer=tf.random_normal_initializer())
    return tf.matmul(inputs, weights) + biases

def full_connect_relu(inputs, weights_shape, biases_shape):
    return tf.nn.relu(full_connect(inputs, weights_shape, biases_shape))

def deep_inference(inputs):
    with tf.variable_scope("layer1"):
        layer = full_connect_relu(inputs, 
                                  [input_units, hidden1_units],
                                [hidden1_units])
    pass