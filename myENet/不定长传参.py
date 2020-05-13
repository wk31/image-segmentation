# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:40:06 2019

@author: w03798
"""

def fun(*args, **kwargs):
    print(args)
    print(kwargs)
    
if __name__ == "__main__":
    fun((1,2))