import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from core.utils import Conv,TransposeConv
from tensorflow.keras.regularizers import l2
from backbone.VGG import VGG

class FCN_vgg16(tf.keras.Model):
    def __init__(self,
                 filters=64,
                 expansion=2,
                 num_classes=2,
                 ):
        super(FCN_vgg16, self).__init__()

        self.backbone = VGG(filters,expansion,num_classes)

        self.skip2 = Conv(num_classes,3)

        self.skip1 = Conv(num_classes,3)

        self.up3 = TransposeConv(num_classes,4)

        self.up2 = TransposeConv(num_classes,4)

        self.up1 = TransposeConv(num_classes,16,strides=8)

        # self.conv = Conv(filters,3)
        # self.classifier = Conv(num_classes,1,kernel_regularizer=l2())

    def call(self, inputs, training=None, mask=None):

        out_1,out_2,out_3=self.backbone(inputs)

        x = self.up3(out_3)

        skip2 = self.skip2(out_2)
        # x = tf.concat([x,skip2],axis=-1)
        x = tf.add_n([x,skip2])

        x = self.up2(x)

        skip1 = self.skip1(out_1)
        # x = tf.concat([x,skip1],axis=-1)
        x = tf.add_n([x,skip1])
        out = self.up1(x)
        # x = self.conv(x)
        # out = self.classifier(x)
        return out