import tensorflow as tf

from tensorflow.keras.layers import (

    BatchNormalization
)
from core.utils import Conv, Aspp, SeparableConv, TransposeConv
from tensorflow.keras.regularizers import l2

class normconnection(tf.keras.Model):
    def __init__(self,filters=64,out_c=2):
        super(normconnection, self).__init__()
        self.conv1 =Conv(filters,3)
        self.bn1 = BatchNormalization()

        self.conv2 =Conv(filters,3)
        self.bn2 =BatchNormalization()
        self.relu= tf.nn.relu

        self.conv3 = Conv(out_c,1,kernel_regularizer=l2())

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x =self.relu(x)

        return self.conv3(x)

class MSC(tf.keras.Model):
    """
    The multi-scale connection.
    Which composed of three branches, each branch have two cascade
    separable convolution( size of 1*(k+i*2) followed by (k+i*2)*1),
    And an element-wise maximum operation is performed to get final features.
    """
    def __init__(self, k=1, filters=64, out_c=2 ):
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
        self.conv = Conv(out_c,1,kernel_regularizer=l2(0.01))

    def call(self, x, training=None, mask=None):
        outputs = []
        for i in range(3):
            l = self.separable1[i](x)
            l = self.bn1[i](l)
            l = self.relu(l)
            l = self.separable2[i](l)
            l = self.bn2[i](l)
            outputs.append(l)
        x = self.relu(tf.maximum(tf.maximum(outputs[0], outputs[1]), outputs[2]))
        x = self.conv(x)
        return x


class Upsample(tf.keras.Model):
    """
    The upsample way in bottom-up path,
    which composed of deconv-conv.
    """
    def __init__(self, filters=64):
        super(Upsample, self).__init__()
        self.Tconv = TransposeConv(filters, 3, strides=2)
        # self.conv = Conv(filters, 3)
        self.Tbn = BatchNormalization()
        self.bn = BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=None, mask=None):
        # x = self.conv(x)
        # x = self.bn(x)
        # x = self.relu(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.Tconv(x)
        x = self.Tbn(x)
        return self.relu(x)
