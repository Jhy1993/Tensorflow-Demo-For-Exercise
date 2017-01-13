# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:24:18 2016

@author: Jhy
"""

import tensorflow as tf
import math
import os
import numpy as np 

# Define
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', None, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 1024,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("thread_number", 1, "Number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "indicates min_after_dequeue of shuffle queue")
flags.DEFINE_string("output_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("model", "deep",
                    "Model to train, option model: deep, linear")
flags.DEFINE_string("optimizer", "sgd", "optimizer to import")
flags.DEFINE_integer('hidden1', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 20, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('steps_to_validate', 10,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train",
                    "Option mode: train, train_from_scratch, inference")
# For distributed
