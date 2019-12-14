import tensorflow as tf
import numpy as np
import os

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss = bce([0.,1.], [1.,1.])
loss2 = tf.nn.sigmoid_cross_entropy_with_logits([0.,1.],[1.,1.])
print('Loss: ', loss.numpy())  # Loss: 11.522857
print(tf.reduce_mean(loss2.numpy()))