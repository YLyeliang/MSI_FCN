import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from core.utils import Conv,TransposeConv
from tensorflow.keras.regularizers import l2
from backbone import DenseNet_MSI
from BottomUp import MSC,Upsample

# def dense_block(x,growth_rate,filters):
# model = DCU(16, 10)
# input = tf.keras.Input(shape=(128, 128, 3))
# t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
# shape = tf.shape(t)[:-1]  # [2, 2, 3]
# print(shape // 2)
# for i in range(5):
#     print(2 ** i)

# _ = model(input)
# model.summary()
# tf.keras.utils.plot_model(model)

class MSI_FCN(tf.keras.Model):
    """
    The architecutre of the proposed FCN with multi-scale input,

    """
    def __init__(self,input_scales=4,
                 dcu_gr=16,
                 dense_gr=24,
                 filters=64,
                 expansion=2,
                 msc_filters=[2,2,2,2],
                 k=(5,3,1,1),
                 aspp_filters=16,
                 up_filters=64,
                 dense_block='bottleneck',
                 num_layers=(4,4,4,4),
                 use_aspp=False,
                 num_classes=2,
                 display_stages=False):
        super(MSI_FCN, self).__init__()
        self.dispaly_stages=display_stages
        self.backbone = DenseNet_MSI(input_scales=input_scales, dcu_gr=dcu_gr, dense_gr=dense_gr,
                                     filters=filters, expansion=expansion,aspp_filters=aspp_filters,
                                     use_aspp=False, dense_block=dense_block, num_layers=num_layers,
                                     display_stages=display_stages)
        self.msc=[]
        for i in range(len(k)): # 64 128 256 512
            self.msc.append(MSC(k[i],filters=msc_filters[i]))

        # deconv of the last layer in top-down.
        self.last_deconv = TransposeConv(num_classes, size=3, strides=2)

        self.upsample=[]
        up_filters=[up_filters for i in range(3)]
        for i in range(3):  # 64 128 256
            self.upsample.append(Upsample(filters=up_filters[i]))

        self.conv=[]
        for i in range(3):
            self.conv.append(Conv(num_classes,1,kernel_regularizer=l2(0.01)))

        self.classifer = Conv(num_classes,size=1,kernel_regularizer=l2(0.01))

    def call(self, x, training=None, mask=None):
        # top-down feature extraction path
        stage_out=self.backbone(x)
        msc_out = []
        # multi-scale connection
        for i in range(len(self.msc)):
            msc_out.append(self.msc[i](stage_out[i]))

        # bottom-up recovery path
        last_up=self.last_deconv(stage_out[-1])
        l = tf.concat([last_up,msc_out[-1]],axis=-1)
        up4=self.conv[2](l)

        l = self.upsample[2](up4)
        l = tf.concat([l,msc_out[2]],axis=-1)
        up3 = self.conv[1](l)

        l = self.upsample[1](up3)
        l = tf.concat([l,msc_out[1]],axis=-1)
        up2 = self.conv[0](l)

        l = self.upsample[0](up2)
        l = tf.concat([l,msc_out[0]],axis=-1)
        out = self.classifer(l)
        if self.dispaly_stages:
            return out,up2,up3,up4

        return out

# inp = Input(shape=[512,512,3],name='input_image')
#
#
# msi_fcn = MSI_FCN()
# SCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)


# x = tf.random.uniform([1,512,512,3])
# with tf.GradientTape() as t:
#     out = msi_fcn(x,training=True)
# # dense_net = msi_fcn.get_layer(name='dense_net_msi').get_layer("dense_block_3")
# dense_net = msi_fcn.get_layer(name='dense_net_msi')
# for layer in dense_net.layers:
#     print(layer.values)
# # gradients = t.gradient(out,msi_fcn.get_layer("dense_net_msi"))
# print()
# print(msi_fcn.summary())


# def msi_fcn(input_size=512, scale=4):
#     sizes = [input_size // (2 ** i) for i in range(4)]
#     inputs = [tf.keras.layers.Input(shape=[s, s, 3]) for s in sizes]
#     assert len(inputs) == scale

# def train_step(model,input,label,epoch):
#     with tf.GradientTape() as t:
#         output = model(input)
#         loss = SCE(output,label)
#     gradients = t.gradient(loss,msi_fcn.trainable_variables)




