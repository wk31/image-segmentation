# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:34:26 2020

@author: w03798
"""

def func(**kwargs):
    print(kwargs)
    for key in kwargs.keys():
        print(key)
        print(kwargs[key])

dict={"name":"wangkuan","age":"18"}
func(**dict)

def func2(*num):
    print(num)

a=(x for x in range(10))
print(a.next())
func2(a)
