import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
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
from core import Conv,TransposeConv
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
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
                 growth_rate=16,
                 filters=64,
                 expansion=2,
                 k=(7,5,3,1),
                 num_layers=(4,4,4,4),
                 num_classes=2):
        super(MSI_FCN, self).__init__()
        self.backbone = DenseNet_MSI(input_scales=input_scales,growth_rate=growth_rate,filters=filters,
                                     expansion=expansion,num_layers=num_layers)
        self.msc=[]
        msc_filters=[filters,filters*2,filters*2,filters*2]
        for i in range(len(k)): # 64 128 256 512
            self.msc.append(MSC(k[i],filters=msc_filters[i]))

        self.aspp_deconv = TransposeConv(filters * 4, size=3, strides=2)

        self.upsample=[]
        up_filters=[filters*2**i for i in range(3)]
        for i in range(3):  # 64 128 256
            self.upsample.append(Upsample(filters=up_filters[i]))

        self.classifer = Conv(num_classes,size=1)
    def call(self, x, training=None, mask=None):
        # top-down feature extraction path
        stage_out=self.backbone(x)
        msc_out = []
        # multi-scale connection
        for i in range(len(self.msc)):
            msc_out.append(self.msc[i](stage_out[i]))

        # bottom-up recovery path
        aspp_upsample=self.aspp_deconv(stage_out[-1])
        l = tf.concat([aspp_upsample,msc_out[-1]],axis=-1)

        l = self.upsample[2](l)
        l = tf.concat([l,msc_out[2]],axis=-1)

        l = self.upsample[1](l)
        l = tf.concat([l,msc_out[1]],axis=-1)

        l = self.upsample[0](l)
        l = tf.concat([l,msc_out[0]],axis=-1)

        l = self.classifer(l)

        return l

inp = Input(shape=[512,512,3],name='input_image')


msi_fcn = MSI_FCN()
SCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)


x = tf.random.uniform([1,512,512,3])
with tf.GradientTape() as t:
    out = msi_fcn(x,training=True)
# dense_net = msi_fcn.get_layer(name='dense_net_msi').get_layer("dense_block_3")
dense_net = msi_fcn.get_layer(name='dense_net_msi')
for layer in dense_net.layers:
    print(layer.values)
# gradients = t.gradient(out,msi_fcn.get_layer("dense_net_msi"))
print()
print(msi_fcn.summary())


def msi_fcn(input_size=512, scale=4):
    sizes = [input_size // (2 ** i) for i in range(4)]
    inputs = [tf.keras.layers.Input(shape=[s, s, 3]) for s in sizes]
    assert len(inputs) == scale

def train_step(model,input,label,epoch):
    with tf.GradientTape() as t:
        output = model(input)

        loss = SCE(output,label)
    gradients = t.gradient(loss,msi_fcn.trainable_variables)




