import tensorflow as tf
from backbone.Densenet import DenseNet
import numpy as np
from tensorflow.keras.layers import Input
from core.utils import Conv,TransposeConv
from tensorflow.keras.regularizers import l2
from backbone.Densenet import basic_dense

class FCD(tf.keras.Model):
    def __init__(self,
                 growth_rate=16,
                 td_filters=[48,112,192,304,464,656,896],
                 up_filters=[1088,816,578,384,256],
                 down_layers=[4,5,7,10,12,15],
                 up_layers=[12,10,7,5,4],
                 num_classes=2):
        super(FCD, self).__init__()
        self.backbone = DenseNet(growth_rate=growth_rate,td_filters=td_filters,
                                 down_layers=down_layers,num_classes=num_classes,dropout=True)

        self.upsample=[]
        self.denseblock=[]
        for i in range(len(up_layers)):
            self.upsample.append(TransposeConv(filters=td_filters[-(i+1)],size=3))
            self.denseblock.append(basic_dense(growth_rate=growth_rate,filters=up_filters[i],num_layers=up_layers[i],dropout=True))
        self.classfier = Conv(num_classes,1)

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs,training)

        for i in range(len(self.upsample)):
            x = self.upsample[i](x)
            x = self.denseblock[i](x)

        out = self.classfier(x)

        return out

