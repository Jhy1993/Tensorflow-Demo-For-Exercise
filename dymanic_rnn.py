# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:
dymanic rnn
INPUT:

OUTPUT:

REFERENCE:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
"""
from __future__ import print_function

import tensorflow as tf
import random

class ToySequenceData(object):
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            len = random.randint(min_seq_len, max_seq_len)
            self.seqlen.append(len)
            if random.random() < 0.5:
                # generate linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value]
                      for i in range(rand_start, rand_start+len)]
                # padding for the same length 后面补0
                s += [[0.0] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1.0, 0.0])
            else:
                s = [[float(random.randint(0, max_value))/max_value]
                      for i in range(len)]
                s += [[0.0] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0.0, 1.0])
        self.batch_id = 0
    
    def next(self, batch_size):
        # While dataset reach end, start over
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:
            min(self.batch_id+batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:
            min(self.batch_id+batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[])
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      