# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:53:43 2019

@author: w03798
"""

import tensorflow as tf
learning_rate = 1e-2
def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
#    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", tf.keras.regularizers.l2(1.e-4))
    activation = conv_params.setdefault("activation", "relu")
    
    def f(input):
        conv = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding,
                activation=activation,
                kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)
    return f

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = tf.keras.layers.BatchNormalization(axis=-1)(input)
    return tf.keras.layers.Activation("relu")(norm)

def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", tf.keras.regularizers.l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = tf.keras.backend.int_shape(input)
    residual_shape = tf.keras.backend.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = tf.keras.layers.Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    return tf.keras.layers.add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and  is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input
    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                              strides=init_strides,
                              padding="same",
#                              kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=1,
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=1)(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=1)(conv_3_3)
        return _shortcut(input, residual)
    return f

class ResnetBuilder():
    def build(input, input_shape, num_outputs, block_fn, repetitions): 
#        input = tf.keras.layers.Input(shape=input_shape)

        conv1 = _conv_bn_relu(filters=64, kernel_size=[3,3], strides=(2, 2))(input)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
#            print(i)
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2
        block = _bn_relu(block)
        
        
        block_shape = tf.keras.backend.int_shape(block)
        pool2 = tf.keras.layers.AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                                 strides=(1, 1))(block)
        flatten1 = tf.keras.layers.Flatten()(pool2)
        dense = tf.keras.layers.Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

#        model = tf.keras.Model(inputs=input, outputs=dense)
        return block
        
#        return block
    def build_resnet_50(input, input_shape, num_outputs):
        return ResnetBuilder.build(input, input_shape, num_outputs, bottleneck, [3, 4, 6, 3])
    
    def build_resnet_101(input, input_shape, num_outputs):
        return ResnetBuilder.build(input, input_shape, num_outputs, bottleneck, [3, 4, 23, 3])
    
    def build_resnet_152(input, input_shape, num_outputs):
        return ResnetBuilder.build(input, input_shape, num_outputs, bottleneck, [3, 8, 36, 3])



def interp_block(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == [473, 473]:
        kernel_strides_map = {1: 60,
                              2: 30,
                              3: 20,
                              6: 10}
    elif input_shape == [713, 713]:
        kernel_strides_map = {1: 90,
                              2: 45,
                              3: 30,
                              6: 15}
    else:
        print("Pooling parameters for input shape ",
              input_shape, " are not defined.")
        exit(1)

    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    prev_layer = tf.keras.layers.AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                        use_bias=False)(prev_layer)
    prev_layer = tf.keras.layers.BatchNormalization(name=names[1])(prev_layer)
    prev_layer = tf.keras.layers.Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
#    prev_layer = Interp(feature_map_shape)(prev_layer)
    prev_layer = tf.image.resize(prev_layer, [60,60], method=tf.image.ResizeMethod.BILINEAR)
    return prev_layer


def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(tf.math.ceil(input_dim / 8.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %
          (feature_map_size, ))

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = tf.keras.layers.Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res



def build_pspnet(nb_classes, resnet_layers=101, input_shape=[473,473,3], activation='softmax'):
    """Build PSPNet."""
    print("Building a PSPNet based on ResNet %i expecting inputs of shape %s predicting %i classes" % (
        resnet_layers, input_shape, nb_classes))
    
    inp = tf.keras.layers.Input((input_shape[0], input_shape[1], 3))
    res=ResnetBuilder.build_resnet_101(inp, input_shape, 21)
    print("Resnet创建完毕")
    psp=build_pyramid_pooling_module(res,[473,473])
    print("psp创建完毕")
    x = _conv_bn_relu(filters=512, kernel_size=[3,3], strides=(1, 1))(psp)
    
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(nb_classes, (1, 1), strides=(1, 1))(x)
    x = tf.keras.layers.Softmax()(x)
    x = tf.image.resize(x, [473,473], method=tf.image.ResizeMethod.BILINEAR)

    model = tf.keras.Model(inputs=inp, outputs=x)

    sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model

























        