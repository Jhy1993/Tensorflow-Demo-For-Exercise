# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 20:36:52 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

CNN + LSTM
"""

import tensorflow as tf 
import numpy as np 

class Config(object):
    """docstring for Config"""
    img_dim = 1024
    hidden_dim = embed_dim = 512
    max_epochs = 50
    batch_size = 256
    keep_prob = 0.75
    layers = 3
    model_name = 'model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d' % (keep_prob, batch_size, hidden_dim, embed_dim, layers)

class Model(object):
    """docstring for Model"""
    def __init__(self, config):
        self.config = config 
        self.load_data()
        self.vocab_size = len(self.index2token)

        #placeholder
        self._sent_placeholder = tf.placeholder(tf.int32, shape=[self.config.batch_size, None], name='sent_ph')
        self._img_placeholer = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.img_dim], name='img_ph')
        self._targets_placeholder = tf.placeholder(tf.int32, shape=[self.config.batch_size, None], name='targets')
        self._dropout_placeholder = tf.placeholder(tf.float32, name='dropout_placeholder')
        
