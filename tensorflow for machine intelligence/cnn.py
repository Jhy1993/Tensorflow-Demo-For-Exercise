# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 22:21:00 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:
 
Reference:
TensorFlow For Machine Intelligence 
"""
import tensorflow as tf 

image_batch = tf.constant([
                            [ # First Image
                            [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
                            [[0, 255, 0], [0, 255, 0], [0, 255, 0]]
                            ],
                            [ # Second Image
                            [[0, 0, 255], [0, 0, 255], [0, 0, 255]],
                            [[0, 0, 255], [0, 0, 255], [0, 0, 255]]
                            ]
                            ])
image_batch.get_shape()



