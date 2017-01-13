# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:48:39 2016

@author: Jhy
https://github.com/tobegit3hub/deep_recommend_system/blob/master/cancer_classifier.py
"""

import datetime
import json
import math
import numpy as np
import os
import tensorflow as tf

# Define 
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', None, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 1024,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("validate_batch_size", 1024,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("thread_number", 1, "Number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "indicates min_after_dequeue of shuffle queue")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("model", "wide_and_deep",
                    "Model to train, option model: wide, deep, wide_and_deep")
flags.DEFINE_string("optimizer", "adagrad", "optimizer to train")
flags.DEFINE_integer('steps_to_validate', 100,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train",
                    "Option mode: train, train_from_scratch, inference")

FEATURE_SIZE = 9
LABEL_SIZE = 2
learning_rate = FLAGS.learning_rate
epoch_number = FLAGS.epoch_number
thread_number = FLAGS.thread_number
batch_size = FLAGS.batch_size
validate_batch_size = FLAGS.validate_batch_size
min_after_dequeue = FLAGS.min_after_dequeue
capacity = thread_number * batch_size + min_after_dequeue
mode = FLAGS.mode
checkpoint_dir = FLAGS.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.mkdirs(checkpoint_dir)
tensorboard_dir = FLAGS.tensorboard_dir
if not os.path.exists(tensorboard_dir):
    os.mkdirs(tensorboard_dir)

# Read TFRecords examples from filename queue
def read_and_decode(filename_queue):
    reader = tf.TFRecorder()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                            'label': tf.FixedLenFeature([], tf.float32),
                                            'features': tf.FixedLenFeature([FEATURE_SIZE], tf.float32)
                                       })
    label = features['label']
    features = features['features']
    return label, features

# Read TFRecord fiels to training
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("data/cancer_train.csv.tfrecords"),
                                                num_epcohs=epoch_number)
label, features = read_and_decode(filename_queue)
batch_labels, batch_features = tf.train.shuffle_batch([label, features],
                                                      batch_size=batch_size,
                                                      num_threads=thread_number,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
# Read TFRecord file for validation
validate_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("data/cancer_test.csv.tfrecords"),
    num_epochs=epoch_number)
validate_label, validate_features = read_and_decode(validate_filename_queue)
validate_batch_labels, validate_batch_features = tf.train.shuffle_batch(
    [validate_label, validate_features],
    batch_size=validate_batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)

# Define the model
input_units = FEATURE_SIZE
hidden1_units = 10
hidden2_units = 10
hidden3_units = 10
hidden4_units = 10
output_units = LABEL_SIZE

def full_connect_relu(inputs, weigths_shape, biases_shape):
    with tf.device('/cpu:0'):
        weigths = tf.get_variable('weights',
                                  weigths_shape,
                                  initializer=tf.random_normal_initializer())
        biases = tf.get_variable('biases',
                                 biases_shape,
                                 initializer=tf.random_normal_intiializer())
        return tf.nn.relu(f.matmul(inputs, weights) + biases)

def deep_inference(inputs):

    pass