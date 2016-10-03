# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 22:01:00 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

3.2 variable
http://tensorfly.cn/tfdoc/how_tos/variables.html
"""

weights = tf.Variable(tf.random_normal([784, 200], stddev=0.25), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

w2 = tf.Variable(weights.initialzed_value() * 2, name="w2")

init_op = tf.initialize_all_variables()
#Add ops to save and restore all variables
saver = tf.train.Server()

with tf.Session() as sess:
    sess.run(init_op)

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print "model save in file: ", save_path
    
