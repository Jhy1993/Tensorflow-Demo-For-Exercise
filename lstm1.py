# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 20:49:20 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
"""

def lstm_model(time_steps, rnn_layers, dense_layers=None):
    """
Creates a deep model based on:
     * stacked lstm cells
     * an optional dense layers
:param time_steps: the number of time steps the model will be looking at.
:param rnn_layers: list of int or dict
                     * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                     * list of dict: [{steps: int, keep_prob: int}, ...]
:param dense_layers: list of nodes for each layer
:return: the model definition
    """
    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            
        pass
    pass
