# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:14:17 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:

Reference:

"""
a = [1, 2, 3]
b = [aa * 10 
        for aa in a]
for pos, val in enumerate(b):
    print('{}=={}'.format(pos, val))

for i in range(len(a)):
    a.pop()
    print a, len(a)