# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:06:11 2019

@author: w03798
"""

import tensorflow as tf
import numpy as np
from dataReader import Config
from dataReader import data_generator
import PIL
from numpy import *
dropout_rate=0.1



config = Config()
data = data_generator(config)
it=iter(data)


def initial_block(inputs, is_training=True):
    net_conv=tf.keras.layers.Conv2D(13, 3, strides=(2,2), padding='same')(inputs)
    net_conv=tf.keras.layers.BatchNormalization()(net_conv)
#    net_conv=tf.keras.layers.Activation(tf.keras.layers.PReLU())(net_conv) 
    net_conv=tf.keras.layers.PReLU()(net_conv)
    
    net_pool=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(inputs)    
    net_concat=tf.keras.layers.concatenate([net_conv,net_pool])

    return net_concat

def bottleneck(inputs, output_depth, internal_scale=4, asymmetric=0, dilated=0, downsample=0, pooling_indices=None,dropout_rate=0.1):        
    if downsample:
        strides=2
    else:
        strides=1
    internel=output_depth//4
    encoder=inputs
    encoder=tf.keras.layers.Conv2D(internel, strides, strides=(strides,strides))(encoder)
    encoder=tf.keras.layers.BatchNormalization(momentum=0.1)(encoder)
#    encoder=tf.keras.layers.Activation(tf.keras.layers.PReLU())(encoder)
    encoder=tf.keras.layers.PReLU()(encoder)
    
    if not asymmetric and not dilated:
        encoder=tf.keras.layers.Conv2D(internel, 3, padding='same')(encoder)
    elif asymmetric: # 卷积拆分 nxn-->1xn + nx1
        encoder = tf.keras.layers.Conv2D(internel, kernel_size=(1, int(asymmetric)), padding='same')(encoder)
        encoder = tf.keras.layers.Conv2D(internel, kernel_size=(int(asymmetric), 1), padding='same')(encoder)
    elif dilated:  # 空洞卷积
        encoder = tf.keras.layers.Conv2D(internel, 3, dilation_rate=(dilated, dilated),padding='same')(encoder)
    else:
        raise(Exception('You shouldn\'t be here'))       
    encoder = tf.keras.layers.BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
#    encoder = tf.keras.layers.Activation(tf.keras.layers.PReLU())(encoder)
    encoder=tf.keras.layers.PReLU()(encoder)
   

    encoder=tf.keras.layers.Conv2D(output_depth, 1, strides=(1,1), padding='same')(encoder)
    encoder=tf.keras.layers.BatchNormalization(momentum=0.1)(encoder)
#    encoder=tf.keras.layers.Activation(tf.keras.layers.PReLU())(encoder)
    encoder=tf.keras.layers.Dropout(rate=dropout_rate)(encoder)
#    print("1---")
    ################################################################
    #other支线
    other= inputs
    if downsample:  # 如果是下采样(只有下采样，通道数才会变化)
#        print("222")
        other, pooling_indices=tf.nn.max_pool_with_argmax(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        other = tf.keras.layers.Permute((1, 3, 2))(other)
        pad_feature_maps = output_depth - inputs.get_shape().as_list()[3]
        tb_pad = (0, 0) # 填充feature map
        lr_pad = (0, pad_feature_maps)  # 填充通道数
        other = tf.keras.layers.ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = tf.keras.layers.Permute((1, 3, 2))(other)
        encoder = tf.keras.layers.add([encoder, other]) # 残差融合
#        encoder = tf.keras.layers.Activation(tf.keras.layers.PReLU())(encoder)
        encoder=tf.keras.layers.PReLU()(encoder)
        return encoder, pooling_indices
    else:
        encoder = tf.keras.layers.add([encoder, other]) # 残差融合
#        encoder = tf.keras.layers.Activation(tf.keras.layers.PReLU())(encoder)
        encoder=tf.keras.layers.PReLU()(encoder)
        return encoder
    
def unbottleneck(inputs, output_depth,  upsample=False, reverse_module=False, dropout_rate=0.01):

    strides = 1
    internel = output_depth//4

    x=inputs
    x=tf.keras.layers.Conv2D(internel, strides, strides=(strides,strides), padding='same')(x)
    x=tf.keras.layers.BatchNormalization(momentum=0.1)(x)
#    x=tf.keras.layers.Activation(tf.keras.layers.PReLU())(x)
    x=tf.keras.layers.PReLU()(x)
    

    if upsample:
        x=tf.keras.layers.Conv2DTranspose(filters=internel, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    else:
        x=tf.keras.layers.Conv2D(internel, 3, padding='same')(x)  
    x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)  # enet uses momentum of 0.1, keras default is 0.99
#    x = tf.keras.layers.Activation(tf.keras.layers.PReLU())(x)
    x=tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(output_depth, strides, strides=(strides,strides), padding='same')(x)
    
    
    
    other = inputs
    if inputs.get_shape()[-1] != output_depth or upsample:
        other = tf.keras.layers.Conv2D(output_depth, strides, strides=(strides,strides), padding='same')(other)
        other =  tf.keras.layers.BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other =  tf.keras.layers.UpSampling2D(size=(2, 2))(other)
        
    if upsample and reverse_module is False:
        decoder = x
    else:
       x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)
#       x = tf.keras.layers.Activation(tf.keras.layers.PReLU())(x)
       x=tf.keras.layers.PReLU()(x)
       decoder = tf.add(x, other)
#       decoder = tf.keras.layers.Activation(tf.keras.layers.PReLU())(decoder)
       decoder=tf.keras.layers.PReLU()(decoder)
    return decoder

    


def buildENet(nc):
    inp = tf.keras.layers.Input((224,224, 3))
    ini = initial_block(inp)
    enet, max_indices1 = bottleneck(ini, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for _ in range(4):
        enet = bottleneck(enet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i
        
    enet, max_indices2 = bottleneck(enet, 128, downsample=True)

    for _ in range(2):
        enet = bottleneck(enet, 128)  # bottleneck 2.1
        enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
        enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
        enet = bottleneck(enet, 128)  # bottleneck 2.5
        enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
        enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8
########################编码结束解码开始###############################
    enet = unbottleneck(enet, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    enet = unbottleneck(enet, 64)  # bottleneck 4.1
    enet = unbottleneck(enet, 64)  # bottleneck 4.2
    enet = unbottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = unbottleneck(enet, 16)  # bottleneck 5.1
 
    enet = tf.keras.layers.Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    enet = tf.keras.layers.Softmax()(enet)
#
#    enet = tf.keras.layers.Conv2D(21, 1, strides=(1,1), padding='same')(enet)
#    enet = tf.keras.layers.Softmax()(enet)
    model = tf.keras.Model(inputs=inp, outputs=enet)

    return model
model=buildENet(21)
#print(model.summary())
sgd = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#############训练过程###################
#for i in range(10):
#    print(i)
#    model.load_weights('./enet/model')
#    model.fit(data, epochs=3, steps_per_epoch=1000,
#              validation_data=data, validation_steps=1)
#    model.save_weights('./enet/model')  
 
    
#################查看预测结果##########
model.load_weights('./enet/model')
for i in range(1):
    X,y=it.get_next()
    X=X[np.newaxis,:]
        
    predictions = model.predict(X)
    a=np.squeeze(predictions)
    y=tf.argmax(predictions,3)
    y=np.squeeze(y)
    y=y*10+30
    img=PIL.Image.fromarray(np.uint8(y))
    img.show()
    
    X=np.squeeze(X)*255
    img1=PIL.Image.fromarray(np.uint8(X))
    img1.show() 


    
    
    

                