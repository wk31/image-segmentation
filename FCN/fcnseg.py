# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:20:13 2019

@author: w03798
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from dataReader import Config
from dataReader import data_generator
import numpy as np
import PIL

from fcn8s import FCN8s
batch_size = 8
num_preprocess_threads=1
min_queue_examples=8

#######数据读取################
config = Config()
data = data_generator(config)
it = iter(data)

class Seg():
    def __init__(self):
        self.model = FCN8s(n_class=21)
        
        print("加载模型")
        self.model.load_weights("F:/工作内容/tensorflow/FCN/weights/model")
        print("加载成功")

        
    def infer(self,x):
#######VC程序传来的是二维数组，返回一维数组#######
#        x=x[np.newaxis,:]/255.0    
#        y=self.model.call(x)
#        y=tf.argmax(y,3)
#        c = np.array(y[0]).flatten()
#        return c
        
#####python程序预测下面的main函数用得到#####
        y=self.model.call(x)
        y=tf.argmax(y,3)
        return y

    
if __name__=="__main__":
    print(tf.keras.__version__)
    model=Seg()
    for i in range(1):
        X,y=it.get_next() 
        
        
        
        y_pre = model.infer(X)

        X=np.squeeze(X)
        X=X*220
        img1 = PIL.Image.fromarray(np.uint8(X))
        img1.show()        
        img1.save('blur{%d}.jpg'%i, 'jpeg')
        
        y_pre=np.squeeze(y_pre)
        y_pre=y_pre*10+30
        img = PIL.Image.fromarray(np.uint8(y_pre))
        img.save('blurseg{%d}.jpg'%i, 'jpeg')
        img.show()
