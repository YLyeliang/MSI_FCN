"""
This file design models in functional method.( related to sub-class method)
"""
import tensorflow as tf

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


# def msi_fcn(input_size=512, scale=4):
#     sizes = [input_size // (2 ** i) for i in range(4)]
#     inputs = [tf.keras.layers.Input(shape=[s, s, 3]) for s in sizes]
#     assert len(inputs) == scale

# def train_step(model,input,label,epoch):
#     with tf.GradientTape() as t:
#         output = model(input)
#         loss = SCE(output,label)
#     gradients = t.gradient(loss,msi_fcn.trainable_variables)