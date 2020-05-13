# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:28:27 2019

@author: w03798
"""
import tensorflow as tf
from tkinter import _flatten
import numpy as np
import math
from math import ceil
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
NUM_CLASSES=21
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
import PIL

def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def batch_norm_layer(inputT, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                   center=False, updates_collections=None, scope=scope+"_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                   updates_collections=None, center=False, scope=scope+"_bn", reuse = True))

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer
                      
def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
        #在cpu上的权重
        kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')#卷积操作
        #在cpu上的偏置项
        biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    
    init = tf.constant_initializer(value=weights,dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    # sess_temp = tf.InteractiveSession()
    sess_temp = tf.global_variables_initializer()
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv

def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):

        logits = tf.reshape(logits, (-1, num_classes))

        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        # consturct one-hot label array将标签拉伸成1维
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]将标签有多少类，拉伸成多少维
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss

def cal_loss(logits, labels):
    loss_weight = np.array([
      0.4,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974,
      1.0974
#      0.2595,
#      0.1826,
#      4.5640,
#      0.1417,
#      0.9051,
#      0.3826,
#      9.6446,
#      1.8418,
#      0.6823,
#      6.2478,
#      7.3614,
#      1.0974,
#      0.2595,
#      0.1826,
#      4.5640,
#      0.1417,
#      0.9051,
#      0.3826,
#      9.6446,
#      1.8418,
#      0.6823
      ]) # class 0~11

    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)

def inference(images, labels, batch_size, phase_train):
        # norm1局部响应归一化函数
        norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                    name='norm1')
        # conv1卷积层1
        conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
        # pool1池化层1
        pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        
        # conv2卷积层2
        conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")
        # pool2池化层2
        pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        
        # conv3
        conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")
        # pool3
        pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool3')
       
        # conv4
        conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")
        # pool4
        pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    
        """ End of encoder """
        """ start upsample """
        # upsample4
        # Need to change when using different dataset out_w, out_h
        # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
        upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, 45, 60, 64], 2, "up4")
        # decode 4
        conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")
    
        # upsample 3
        # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
        upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, 90, 120, 64], 2, "up3")
        # decode 3
        conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")
    
        # upsample2
        # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
        upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, 180, 240, 64], 2, "up2")
        # decode 2
        conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")
    
        # upsample1
        # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
        upsample1= deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 360, 480, 64], 2, "up1")
        # decode4
        conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
        """ end of Decode """
        """ Start Classify """
        # output predicted class number (6)
        with tf.variable_scope('conv_classifier') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                   shape=[1, 1, 64, NUM_CLASSES],
                                                   initializer=msra_initializer(1, 64),
                                                   wd=0.0005)
            conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
            conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
    
            logit = conv_classifier
            loss = cal_loss(conv_classifier, labels)
    
        return loss, logit
                       
                       
class segTest:                 
    def __init__(self):
        batch_size = 1
        image_h = 360
        image_w = 480
        image_c = 3
        
        self.test_data_node = tf.placeholder(
                tf.float32,shape=[batch_size, image_h, image_w, image_c])
        self.test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, 360, 480, 1])
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.loss, self.logits = inference(self.test_data_node, self.test_labels_node, batch_size, self.phase_train)
        self.pred = tf.argmax(self.logits, axis=3)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        variable_averages = tf.train.ExponentialMovingAverage(
                          MOVING_AVERAGE_DECAY)
        #滑动平均值方式加载模型
        variables_to_restore = variable_averages.variables_to_restore()
    
        self.saver = tf.train.Saver(variables_to_restore)
        self.saver=tf.train.import_meta_graph("C:/Users/w03798/Desktop/Logs/model.ckpt-9000.meta")        
        self.saver.restore(self.sess, "C:/Users/w03798/Desktop/Logs/model.ckpt-9000")
        
        
              
    def predict(self,image_batch):
        label_batch = np.zeros((360,480,1),dtype=np.float32)
        label_batch = label_batch[np.newaxis]
        #print(image_batch)
        imgnew = PIL.Image.fromarray(image_batch)
        imgnew.save('./w1.png')
        image_batch = image_batch[np.newaxis,:]
        print('##########################')
        feed_dict = {
                    self.test_data_node: image_batch,
                    self.test_labels_node: label_batch,
                    self.phase_train: False
                    }

        logit, segimg = self.sess.run([self.logits, self.pred], feed_dict=feed_dict)
        #return logit,segimg
        writeImage(segimg[0], 'ww.png')
        #return(np.array(segimg[0]))
        c = np.array(segimg[0]).flatten()
        return(c)
        #b=list(_flatten(np.array(segimg[0])))
        #b = sum(segimg[0], [])
        #return b
        print('##########################')



if __name__ == '__main__':
    segnet = segTest() 
    #segnet.__init__
    img = PIL.Image.open('D:/DATA/voc2012/VOC2012/JPEGImagesNEW/2007_000333.jpg')
    im = np.array(img, np.float32)
    im = im[np.newaxis]
    b=segnet.predict(im)


    
    
    
    
    