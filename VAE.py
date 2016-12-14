# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:20:02 2016

@author: Jhy_BISTU
README：
VAE: Variational Autoencoder in Tensorflow
INPUT:

OUTPUT:

REFERENCE:
https://jmetzen.github.io/2015-11-27/vae.html
"""

import numpy as np
import tensorflow as tf
import matplotlib.pypolt as plt
import input_data

np.random.seed(0)
tf.set_random_seed(0)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class VariationalAutoencoder(object):
    """docstring for VariationalAutoencoder"""
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # Tf input
        self.x = tf.placeholder(tf.float32, [None, network_architecture['n_input']])
        # Create autoencoder 
        self._create_network()
        self._create_loss_optimizer()
        # Launch the sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def _create_network(self):
        network_weights = self._initialize_weights(**self.network_architecture)
        # Use recongnition network to determine mean and variance 
        # of Gaussian distribution in latent space
        self.z_mean = self._recognition_network(network_weights['weights_recog'])
        self.z_log_sigma_sq = self._recognition_network(network_weights['biases_recog'])
        # Draw one sample z from Gaussian distribution
        n_z = self_network_architecture['n_z']
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)
        #z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.matmul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        # Use generator to detetmine mean of Bernoulli distribution
        # of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
    def _initialize_weights(self, n_hidden_recog_1)
        

        