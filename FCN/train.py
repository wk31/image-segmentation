# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
from dataReader import Config
from dataReader import data_generator
import PIL
import random
from fcn8s import FCN8s
batch_size = 2
num_preprocess_threads=1
min_queue_examples=8


config = Config()
data = data_generator(config)
it = iter(data)


model = FCN8s(n_class=21)
model.load_weights('./weights/model')
  
learning_rate=0.0001
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)


################训练########################
num_batches=30
for batch_index in range(num_batches):
    X,y=it.get_next()
#    print(X)
    with tf.GradientTape() as tape:
        print(batch_index)
        y_pred=model.call(X)
#        print(y)
#        print(y_pred)
        loss=tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=y_pred)
        loss=tf.reduce_mean(loss)
        if batch_index%1==0:
            print("batch:%d,loss:%f"%(batch_index,loss.numpy()))
    
    grads=tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
    
model.save_weights('./weights/model')



#model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
#             loss=tf.keras.losses.sparse_categorical_crossentropy,
#             metrics=[tf.keras.metrics.categorical_accuracy])
#
#model.fit(data, epochs=3, steps_per_epoch=50,
#          validation_data=data, validation_steps=1)
#model.save_weights('./weights/model')






