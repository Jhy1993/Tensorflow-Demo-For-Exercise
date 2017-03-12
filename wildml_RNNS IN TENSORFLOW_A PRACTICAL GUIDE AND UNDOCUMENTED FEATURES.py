# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:
RNNS IN TENSORFLOW, A PRACTICAL GUIDE AND UNDOCUMENTED FEATURES
INPUT:

OUTPUT:

REFERENCE:
http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
"""
from __future__ import print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

x = np.random.randn(2, 10, 8)
print(x.shape)

x[1, 6:] = 0
x_len = [10, 6]

cell = tf.nn.rnn_cell.LSTMCell(num_units=64,
                               state_is_tuple=True)

cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,
                                     output_keep_prob=0.5)
cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 4,
                                   state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
                                         cell=cell,
                                         dtype=tf.float32,
                                         sequence_length=x_len,
                                         inputs=x)
result = tf.contrib.learn.run_n(
                                {"outputs": outputs,
                                 "last_states": last_states},
                                 n=1,
                                 feed_dict=None)
assert result[0]["outputs"].shape == (2, 10, 64)

assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()
















