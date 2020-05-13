# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:14:25 2019

@author: w03798
"""

import tensorflow as tf
X = tf.constant([[1.0, 2.0, 3.0, 4], [4.0, 5.0, 6.0, 7]])
y = tf.constant([[10.0], [20.0]])

#模型定义
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=tf.zeros_initializer()
                )
        
    def call(self, input):
        output = self.dense(input)
        return output
   
    
model = Linear()
#优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred=model(X)
        #损失函数
        loss=tf.reduce_mean(tf.square(y_pred-y))
        
    grads=tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
    
print(model.variables)