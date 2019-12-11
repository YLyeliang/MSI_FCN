import tensorflow as tf
from tensorflow.keras.layers import Conv2D

def Conv(filters, size, strides=1, dilation=None, initializer=tf.keras.initializers.he_normal()):
    if strides == 1:
        padding = 'same'
    else:
        padding = 'valid'
    if dilation is None:
        dilation = (1, 1)
    conv = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding,
                  dilation_rate=dilation, kernel_initializer=initializer)
    return conv

def SeparableConv(filters,size,strides=1,depth_multiplier=1,initializer=tf.keras.initializers.he_normal()):
    if strides == 1:
        padding ='same'
    else:
        padding = 'valid'

    separable_conv = tf.keras.layers.SeparableConv2D(filters=filters,kernel_size=size,strides=strides,
                                                     padding=padding,depth_multiplier=depth_multiplier,
                                                     depthwise_initializer=initializer,
                                                     pointwise_initializer=initializer)
    return separable_conv

def TransposeConv(filters,size,strides=1,initializer = tf.keras.initializers.glorot_uniform()):
    """
    Define a Deconv.
    """
    transpose_conv = tf.keras.layers.Conv2DTranspose(filters,kernel_size=size,strides=strides,
                                                     padding='same',kernel_initializer=initializer)
    return transpose_conv



class Aspp(tf.keras.Model):
    """
    The definition of atrous spatial pyramid pooling.
    Which composed of part a): four convolution with different dilation rate.
    part b): an image-level feature pooling.
    And concat them together, followed by a 1 * 1 convolution.
    """
    def __init__(self, filters=128):
        super(Aspp, self).__init__()
        self.atrous_conv1 = Conv(filters, 1)
        self.atrous_conv2 = Conv(filters, 3, dilation=6)
        self.atrous_conv3 = Conv(filters, 3, dilation=12)
        self.atrous_conv4 = Conv(filters, 3, dilation=18)

        # image level features
        self.conv1 = Conv(filters, 1)

        # conv 1*1 after concat
        self.conv2 = Conv(filters*4, 1)

    def call(self, x):
        input_size = tf.shape(x)[1:3]
        block1 = self.atrous_conv1(x)
        block2 = self.atrous_conv2(x)
        block3 = self.atrous_conv3(x)
        block4 = self.atrous_conv4(x)

        # image level features
        # global average pooling
        features = tf.reduce_mean(x, [1, 2], keepdims=True)
        features = self.conv1(features)
        features = tf.image.resize(features, input_size)

        return self.conv2(tf.concat([block1, block2, block3, block4, features], axis=-1))

# class
