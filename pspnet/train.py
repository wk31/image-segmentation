# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:47:52 2019

@author: w03798
"""

from myresnet import build_pspnet
import tensorflow as tf
import numpy as np
from dataReader import data_generator
from dataReader import Config
import PIL


#图像生成
config = Config()
data = data_generator(config)
model=build_pspnet(21)

################训练用###########################
for i in range(30):
    print(i)
    model.load_weights('./psp101/model') 
    model.fit(data, epochs=3, steps_per_epoch=1000,
          validation_data=data, validation_steps=1)
    model.save_weights('./psp101/model') 



################查看分割结果用####################
#model.load_weights('./psp101/model')      
#for i in range(10):
#    X,y=it.get_next()     
#    X=X[np.newaxis,:] 
#    predictions = model.predict(X)
#    
#    y=tf.argmax(predictions,3)
#    y=np.squeeze(y)
#    y=y*10
#    img=PIL.Image.fromarray(np.uint8(y))
#    img.show()
#    
#    X=np.squeeze(X)*255
#    img1=PIL.Image.fromarray(np.uint8(X))
#    img1.show()    

