# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:00:47 2019

@author: w03798
"""

import tensorflow as tf

class Config(object):
    TRAIN_NAME_PATH = 'D:/DATA/voc2012/VOC2012/ImageSets/Segmentation/train.txt'
    TRAIN_PIC_PATH = 'D:/DATA/voc2012/VOC2012/JPEGImagesNEW/'
    TRAIN_SEG_PATH = 'D:/DATA/voc2012/VOC2012/SegmentationClass_augNEW/'
    batch_size = 2


def _decode_and_resize(image_name, label_name):
    print("aaa")
    image_string = tf.io.read_file(image_name)
    image_decode = tf.image.decode_jpeg(image_string)
    image_converted = tf.image.resize(image_decode, [473, 473])/255.0
    
    label_string = tf.io.read_file(label_name)
    label_decoded = tf.image.decode_png(label_string)
    label_converted = tf.image.resize(label_decoded, [473, 473], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
    return image_converted, label_converted


def get_filename_list(config):
    fd = open(config.TRAIN_NAME_PATH)
    image_filenames = []
    label_filenames = []
    for i in fd:
        i = i.replace('\n','')
        image_filenames.append(config.TRAIN_PIC_PATH+i+'.jpg')
        label_filenames.append(config.TRAIN_SEG_PATH+i+'.png')
    return image_filenames, label_filenames


def image_label_path_generator(config):
    image_filenames, label_filenames=get_filename_list(config)
    datasetname = tf.data.Dataset.from_tensor_slices((image_filenames,label_filenames))
    datasetname=datasetname.shuffle(1000)
    return datasetname
    
    

#######数据生成########
def data_generator(config):
#    datasetname=image_label_path_generator('D:/DATA/voc2012/VOC2012/ImageSets/Segmentation/train.txt')
    datasetname=image_label_path_generator(config)
    dataset = datasetname.map(_decode_and_resize)
    dataset = dataset.repeat().shuffle(500)
    data=dataset.batch(config.batch_size)
    it = iter(dataset)
    return data

if __name__ == "__main__":
    config = Config()
    data = data_generator(config)
    print(data)