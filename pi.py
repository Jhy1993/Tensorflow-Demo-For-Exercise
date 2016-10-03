# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:16:38 2016

@author: Jhy1993
"""
import tensorflow as tf
import matplotlib.pyplt as plt
#------------------------------
# x = tf.placeholer(tf.float32)
# y = 2 * x * x

# var_grad = tf.gradients(y, x)
# with tf.Session() as sess:
#     var_grad_val = sess.run(var_grad, feed_dict={x: 1.})
#     print (var_grad_val)

# # return [4]

# #-------------------------------------------
# uniform = tf.random_uniform([100], minval=0, maxval=1, dtype=tf.float32)

# with tf.Session() as sess:
#     print uniform.eval()
#     plt.hist(uniform_eval(), normed=True)
#     plt.show()

#--------------计算pi-------
import tensorflow as tf 
trials = 100
hits = 0

x = tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float32)
y = tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float32)
pi = []

with tf.Session().as_default():
    for i in range(1, trials):
        for j in range(1, trials):
            if x.eval() ** 2 + y.eval() **2 < 1:
                hits = hits + 1
                pi.append((4 * float(hits) / i) / trials)
plt.plot(pi)
plt.show()


