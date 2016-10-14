# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 12:54:33 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:
但是当创建复杂的模块时，通常你需要共享大量变量集并且如果你还想在同一个地方初始化这所有的变量,
我们又该怎么做呢.本教程就是演示如何使用tf.variable_scope() 
和tf.get_variable()两个方法来实现这一点.
Reference:
http://tensorfly.cn/tfdoc/how_tos/variable_scope.html
"""

def my_image_filter(input_images):
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]), name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
    relu1 = tf.nn.relu(conv1 + conv1_biases)

    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32,32]), name="conv2_wights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    conv2 = tf.nn.conv2d(relu1, conv2_wights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2 + conv2_biases)

#===========================better code======================
def conv_relu(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constan_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        return conv_relu(relu1, [5, 5, 32, 32], [32])

with tf.variable_scope("image_filter") as scope:
    result1 = my_image_filter(image1)
    scope.reuse_variables()
    result2 = my_image_filter(image2)





