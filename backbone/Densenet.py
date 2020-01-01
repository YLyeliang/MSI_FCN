import tensorflow as tf
from tensorflow.keras.layers import (
    MaxPool2D,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
from core.utils import Conv, Aspp

class bottle_dense(tf.keras.Model):
    """
    A standard dense block in DenseNet, which has bottleneck
    """

    def __init__(self, growth_rate, filters, num_layers=4):
        super(bottle_dense, self).__init__()
        self.conv3 = []  # 3*3 conv
        self.conv1 = []  # 1*1 conv
        self.bn = []
        for i in range(num_layers):
            self.conv3.append(Conv(growth_rate, 3))
            self.conv1.append(Conv(growth_rate * 2, 1))
            self.bn.append(BatchNormalization())

        assert len(self.conv3) == len(self.conv1)
        self.transition = Conv(filters, 1)
        self.transBN = BatchNormalization()

        self.relu = tf.nn.relu

    def call(self, x):
        for i in range(len(self.conv3)):
            l = self.conv1[i](x)
            l = self.conv3[i](l)
            l = self.relu(l)
            x = tf.concat([x, l], axis=-1)

        x = self.transition(x)
        x = self.transBN(x)

        return self.relu(x)

class basic_dense(tf.keras.Model):
    def __init__(self, growth_rate, filters, num_layers=4,dropout=False):
        super(basic_dense, self).__init__()
        if dropout is True:
            self.dropout = tf.keras.layers.Dropout(0.2)
        self.conv3 = []  # 3*3 conv
        self.bn = []
        for i in range(num_layers):
            self.conv3.append(Conv(growth_rate, 3))
            self.bn.append(BatchNormalization())

        self.transition = Conv(filters, 1)
        self.transBN = BatchNormalization()

        self.relu = tf.nn.relu

    def call(self, x, training=None, mask=None):
        for i in range(len(self.conv3)):
            l = self.conv3[i](x)
            if self.dropout and training:
                l = self.dropout(l)
            l = self.relu(l)
            x = tf.concat([x, l], axis=-1)

        x = self.transition(x)
        if self.dropout and training:
            x = self.dropout(x)
        x = self.transBN(x)

        return self.relu(x)


class DenseNet(tf.keras.Model):
    """
    The top-down feature extraction path with multi-scale input.
    """

    def __init__(self,
                 growth_rate=16,
                 td_filters=[48, 112, 192, 304, 464, 656,896],
                 down_layers=[4, 5, 7, 10, 12, 15],
                 num_classes=2,
                 dropout=True):
        super(DenseNet, self).__init__()
        self.relu = tf.nn.relu
        self.conv1 = Conv(td_filters[0], 3)
        self.bn1 = BatchNormalization()  # 256 128 64 32

        self.maxpool = MaxPool2D()


        self.denseblock = []

        self.filters = td_filters
        for i in range(len(down_layers)):  # filters: 64 128 256 512
            self.denseblock.append(basic_dense(growth_rate, self.filters[i+1], num_layers=down_layers[i],dropout=dropout))

    def call(self, inputs, training=None, mask=None):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        for i in range(len(self.denseblock)):
            x = self.denseblock[i](x,training)
            if not i == len(self.denseblock)-1:
                x = self.maxpool(x)

        return x


# x = tf.Variable(tf.zeros([1, 512, 512, 3]))
# model = DenseNet_MSI()
# out = model(x)
# print(model.summary())
# tf.keras.utils.plot_model(model,show_shapes=True,dpi=96)
# debug = 1
