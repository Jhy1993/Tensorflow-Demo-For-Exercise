# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:37:12 2016

@author: Jhy1993
"""

import datatime
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import seq2seq
import codecs

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_integer('epoch_number', 10, 'Number of epochs')
flags.DEFINE_integer("batch_size", 32,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("optimizer", "adam", "optimizer to train")
flags.DEFINE_integer('steps_to_validate', 1,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train", "Opetion mode: train, inference")
flags.DEFINE_string("image", "./data/inference/Pikachu.png",
                    "The image to inference")
flags.DEFINE_string("inference_start_word", "l", "The start word to inference")
flags.DEFINE_string(
    "model", "stacked_lstm",
    "Model to train, option model: lstm, bidirectional_lstm, stacked_lstm")


def main():
    print('start generating lycrics')
    
    batch_size = FLAGS.batch_size
    epoch_number = FLAGS.epoch_number
    sequence_length = 20
    rnn_hidden_units = 100
    stacked_layer_number = 3
    
    lycrics_filepath = './data/shakespeare.txt'
    
    f = codecs.open(lycrics_filepath, encoding='utf-8')
    lycrics_data = f.read()
    
    words = list(set(lycrics_data))
    words.sort()
    vocabulary_size = len(words)
    char_id_map = {}
    id_char_map = {}
    for index, char in enumerate(words):
        id_char_map[index] = char
        char_id_map[char] = index


    train_dataset = []
    train_labels = []
    index = 0
    for i in range(batch_size):
        features = lycrics_data[index:index + sequence_length]
        labels = lycrics_data[index + 1:index + sequence_length + 1]
        index += sequence_length

        features = [char_id_map[word] for word in features]
        labels = [char_id_map[word] for word in labels]

        train_dataset.append(features)
        train_labels.append(labels)

    batch_size = FLAGS.batch_size
    mode = FLAGS.mode

    if mode == 'inference':
        batch_size = 1
        sequence_length = 1

    x = tf.placeholder(tf.int32, shape=(None, sequence_length))
    y = tf.placeholder(tf.int32, shape=(None, sequence_length))
    epoch_number = FLAGS.epoch_number
    checkpoint_dir = FLAGS.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    tensorboard_dir = FLAGS.tensorboard_dir

    checkpoint_file = checkpoint_dir + '/checkpoint.ckpt'
    steps_to_validate = FLAGS.steps_to_validate

    def lstm_inference(x):
        pass

    def stacked_lstm_inference(x):
        lstm_cell = rnn_cell.BasicLSTMCell(rnn_hidden_units)
        lstm_cells = rnn_cell.MultiRNNCell([lstm_cell] * stacked_layer_number)
        initial_state = lstm_cells.zero_state(batch_size, tf.float32)

        with tf.variable_scope("stacked_lstm"):
            weights = tf.get_variable("weights",
                                      [rnn_hidden_units, vocabulary_size])
            bias = tf.get_variable("bias", [vocabulary_size])
            embedding = tf.get_variable("embedding",
                                        [vocabulary_size, rnn_hidden_units])

        
        inputs = tf.nn.embedding_lookup(embedding, x)
        outputs, last_state = tf.nn.dynamic_rnn(lstm_cells,
                                                inputs,
                                                initial_state=initial_state)
        outpus = tf.reshape(outputs, [-1, rnn_hidden_units])
        logits = tf.add(tf.matmul(output, weights), bias)

        return logits, lstm_cells, initial_state, last_state

    def inference(inpus):
        print("Use the model: {}".format(FLAGS.model))
        if FLAGS.model == "lstm":
            return lstm_inference(inputs)
        elif FLAGS.model == "stacked_lstm":
            return stacked_lstm_inference(inputs)
        else:
            print("Unkonw model, exit now")
            exit(1)

    # Define train op
    logits, lstm_cells, initial_state, last_state = inference(x)

    targets = tf.rehape(y, [-1])
    

        pass

            )
        pass






    
    
    


















