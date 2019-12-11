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
from core import Conv, Aspp


class DCU(tf.keras.Model):
    """
    This is a dense conv unit, which perform a densely connected conv
    without bottleneck.
    """

    def __init__(self, growth_rate, filters):
        super(DCU, self).__init__()
        self.conv = []
        self.bn = []
        for i in range(4):
            self.conv.append(Conv(growth_rate, 3))
            self.bn.append(BatchNormalization())

        self.transition = Conv(filters, 1)
        self.bn5 = BatchNormalization()

        self.relu = tf.nn.relu

        # self.conv1 = Conv(growth_rate,3)
        # self.bn1 = BatchNormalization()
        #
        # self.conv2 = Conv(growth_rate,3)
        # self.bn2 = BatchNormalization()
        #
        # self.conv3 = Conv(growth_rate,3)
        # self.bn3 = BatchNormalization()
        #
        # self.conv4 = Conv(growth_rate,3)
        # self.bn4 = BatchNormalization()
        #
        # self.conv5 = Conv(filters,1)
        # self.bn5 = BatchNormalization()

    def call(self, x):
        for i in range(len(self.conv)):
            l = self.conv[i](x)
            l = self.bn[i](l)
            l = self.relu(l)
            x = tf.concat([x, l], axis=-1)

        x = self.transition(x)
        x = self.bn5(x)

        return self.relu(x)


class dense_block(tf.keras.Model):
    """
    A standard dense block in DenseNet, which has bottleneck
    """

    def __init__(self, growth_rate, filters, num_layers=4):
        super(dense_block, self).__init__()
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


class DenseNet_MSI(tf.keras.Model):
    """
    The top-down feature extraction path with multi-scale input.
    """

    def __init__(self,
                 input_scales=4,
                 growth_rate=16,
                 filters=64,
                 expansion=2,
                 aspp_filters=64,
                 first_dense=True,
                 num_layers=(4, 4, 4, 4)):
        super(DenseNet_MSI, self).__init__()
        # self.relu = tf.nn.relu
        # self.conv1 = Conv(64, 3)
        # self.bn1 = BatchNormalization()  # 256 128 64 32

        self.maxpool = MaxPool2D()

        self.input_scales = input_scales
        self.dcu = []
        for i in range(input_scales - 1):
            self.dcu.append(DCU(16, 64))

        assert len(self.dcu) == self.input_scales - 1

        self.denseblock = []

        for i in range(len(num_layers)):  # filters: 64 128 256 512
            self.denseblock.append(dense_block(growth_rate, filters, num_layers=num_layers[i]))
            filters *= expansion

        self.aspp = Aspp(filters=aspp_filters)

    def call(self, inputs, training=None, mask=None):
        shape = tf.shape(inputs)[1:3]

        # The list of multi-scale input.
        scales = []
        for i in range(self.input_scales):
            down = 2 ** i
            if i == 0:
                scales.append(inputs)
            else:
                scales.append(tf.image.resize(inputs, shape // down))

        # The list of features extracted from multi-scale input except the biggest image.
        scales_out = []
        for i in range(self.input_scales - 1):
            scales_out.append(self.dcu[i](scales[i + 1]))

        # Output of each stage.
        stage_out = []
        # x = self.conv1(scales[0])
        # x = self.bn1(x)
        # x = tf.nn.relu(x)
        x = self.denseblock[0](scales[0])
        stage_out.append(x)
        x = self.maxpool(x)
        for i in range(1, len(self.denseblock)):
            if i < self.input_scales:
                x = tf.concat([x, scales_out[i - 1]], axis=-1)
            x = self.denseblock[i](x)
            stage_out.append(x)
            x = self.maxpool(x)

        # aspp
        x = self.aspp(x)
        stage_out.append(x)
        return tuple(stage_out)


# x = tf.Variable(tf.zeros([1, 512, 512, 3]))
# model = DenseNet_MSI()
# out = model(x)
# print(model.summary())
# tf.keras.utils.plot_model(model,show_shapes=True,dpi=96)
# debug = 1
