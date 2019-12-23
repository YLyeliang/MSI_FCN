import tensorflow as tf
from tensorflow.keras.layers import (
    MaxPool2D,
    BatchNormalization
)
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from core.utils import Conv, Aspp,conv_block




class VGG(tf.keras.Model):
    def __init__(self,
                 filters,
                 expansion=2,
                 num_classes=2,):
        super(VGG, self).__init__()

        self.filters=filters
        self.conv1 = Sequential(conv_block(self.filters,3,num_layers=2))

        self.conv2 = Sequential(conv_block(self.filters*expansion,3,num_layers=2))

        self.conv3 = Sequential(conv_block(self.filters*(expansion**2),3,num_layers=3))

        self.conv4 = Sequential(conv_block(self.filters*(expansion**3),3,num_layers=3))

        self.conv5 = Sequential(conv_block(self.filters*(expansion**3),3,num_layers=3))

        self.conv6 = Sequential(conv_block(self.filters*(expansion**4),3,num_layers=2,drop_out=True))

        self.conv7 = Conv(self.filters*(expansion**4),3)

        self.maxpool = MaxPool2D([2, 2], 2)

    def call(self, inputs, training=None, mask=None):
        stage_out=[]
        
        x = self.conv1(inputs)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        out1 = self.maxpool(x)

        x = self.conv4(out1)
        out2 = self.maxpool(x)

        x = self.conv5(out2)
        x = self.maxpool(x)

        x = self.conv6(x)
        out3 = self.conv7(x)
        return out1,out2,out3

