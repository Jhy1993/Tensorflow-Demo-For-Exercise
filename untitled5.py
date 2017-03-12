# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:

INPUT:

OUTPUT:

REFERENCE:

"""
def hi():
    return 'hi yas'

def jhy(func):
    print('before do hi')
    print(func())
    
jhy(hi)

def a_new_de(a_func):
    
    def warpTheFunc():
        print('befor run a_func()')
        
        a_func()
        
        print('after run a_func()')
        
    return warpTheFunc

def a_func_need_decoraation



















