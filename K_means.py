# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 20:36:52 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

K means
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

num_vector = 1000
num_clusters = 4
num_step = 100

x_value = []
y_value = []
vector_values = []

for i in xrange(num_vector):
    if np.random.random() > 0.5:
        x_value.append(np.random.normal(0.4, 0.7))
        y_value.append(np.random.normal(0.2, 0.8))
    else:
        x_value.append(np.random.normal(0.6, 0.4))
        y_value.append(np.random.normal(0.8, 0.5))
vector_values = zip(x_value, y_value)
vectors = tf.constant(vector_values)

n_samples = tf.shape(vector_values)[0]
random_indices = tf.random_shuffle(tf.range(0, n_samples))

begin = [0, ]
size = [num_clusters]
size[0] = num_clusters

centroid_indices = tf.slice(random_indices, begin, size)
centroids = tf.Variables(tf.gather(vector_values, centroid_indices))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

vectors_subtration = tf.sub(expanded_vectors,expanded_centroids)

euclidean_distances = tf.reduce_sum(tf.square(vectors_subtration), 2)
assignments = tf.to_int32(tr.argmin(euclidean_distances, 0))

.....................
