# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:45:39 2017

@author: jhy
"""
n = 10
time2 = 0
time3 = 0
time5 = 0
seq = [1]
for i in range(n-1):
    u2 = seq[time2] * 2
    u3 = seq[time3] * 3
    u5 = seq[time5] * 5
    s = min(u2, u3, u5)
    seq.append(s)
    if s == u2:
        time2 += 1
    if s == u3:
        time3 += 1
    if s == u5:
        time5 += 1