# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:11:59 2019

@author: w03798
"""
import tensorflow as tf
print(tf.__version__)
A = tf.constant([[1,2],[3,4]])
B = tf.constant([[5,6],[7,8]])
C = tf.matmul(A,B)
print(C)
print(tf.random.uniform(shape=()))
print(tf.zeros(shape=(2,4)))
a = C.numpy()



x = tf.Variable(initial_value=3.)

with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad = tape.gradient(y,x)
print(y_grad)


import numpy as np
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw-X_raw.min())/(X_raw.max() - X_raw.min())
y = (y_raw-y_raw.min())/(y_raw.max() - y_raw.min())

a, b = 0, 0

num_epoch = 10000
learning_rate = 1e-3
for e in range(num_epoch):
    # 手动计算损失函数关于自变量（模型参数）的梯度
    y_pred = a * X + b
    grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()

    # 更新参数
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print(a, b)









