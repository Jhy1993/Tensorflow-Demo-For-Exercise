# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:

INPUT:

OUTPUT:

REFERENCE:

"""
import numpy as np

a = np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])
print(a)
def shuffle(x):
    permutate = np.random.permutation(x.shape[0])
    print(permutate)
    x_shuffled = x[permutate, :]
    print(x_shuffled)
    
shuffle(a)

a = [[float(i)/10] for i in range(3, 7)]
print(a)
a += [[0.0] for i in range(10)]
print(a)