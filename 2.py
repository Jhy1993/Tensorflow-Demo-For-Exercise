# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:16:38 2016

@author: Jhy1993
"""
import tensorflow as tf
a = tf.constant(10, name="a")
b = tf.constant(90, name="b")
y = tf.Variable(a + b * 2, name="y")
init = tf.initialize_all_variables()
with tf.Session() as sess:
    merges = tf.merge_all_summaries()
    #This instruction must merge all the summaries collected in the default graph.
    writer = tf.train.SummaryWriter("./tflogs", sess.graph)
    sess.run(init)
    print(sess.run(y))
#在命令行输入, 得到端口号,查看
tensorboard --logdir=./tf.logs
 
import numpy as np 
tensor_1d = np.array([1.3, 1., 4.0])
tf.tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)
with tf.Session() as sess:
    print sess.run(tf_tensor)

matrix1 = np.array([(2,2,2),(2,2,2),(2,2,2)],dtype='int32')
matrix2 = np.array([(1,1,1),(1,1,1),(1,1,1)],dtype='int32')

mat1 = tf.constant(matrix1)
mat2 = tf.constant(matrix2)

mat_product = tf.matmul(mat1, mat2)
mat_sum = tf.add(mat1, mat2)
