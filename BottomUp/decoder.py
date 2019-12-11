import tensorflow as tf

from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization
)
from core import Conv, Aspp, SeparableConv, TransposeConv


class MSC(tf.keras.Model):
    """
    The multi-scale connection.
    Which composed of three branches, each branch have two cascade
    separable convolution( size of 1*(k+i*2) followed by (k+i*2)*1),
    And an element-wise maximum operation is performed to get final features.
    """
    def __init__(self, k=1, filters=64, ):
        super(MSC, self).__init__()
        self.separable1 = []
        self.separable2 = []
        self.bn1 = []
        self.bn2 = []
        for i in range(3):
            self.separable1.append(SeparableConv(filters, (1, k + 2 * i)))
            self.bn1.append(BatchNormalization())
            self.separable2.append(SeparableConv(filters, (k + 2 * i, 1)))
            self.bn2.append(BatchNormalization())
        self.relu = tf.nn.relu

    def call(self, x, training=None, mask=None):
        outputs = []
        for i in range(3):
            l = self.separable1[i](x)
            l = self.bn1[i](l)
            l = self.relu(l)
            l = self.separable2[i](l)
            l = self.bn2[i](l)
            outputs.append(l)
        x = tf.maximum(tf.maximum(outputs[0], outputs[1]), outputs[2])
        return self.relu(x)


class Upsample(tf.keras.Model):
    """
    The upsample way in bottom-up path,
    which composed of deconv-conv.
    """
    def __init__(self, filters=64):
        super(Upsample, self).__init__()
        self.Tconv = TransposeConv(filters, 3, strides=2)
        self.conv = Conv(filters, 3)
        self.Tbn = BatchNormalization()
        self.bn = BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=None, mask=None):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.Tconv(x)
        x = self.Tbn(x)
        return self.relu(x)
