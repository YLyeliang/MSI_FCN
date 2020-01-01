import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from core.utils import Conv,TransposeConv
from tensorflow.keras.regularizers import l2
from backbone.VGG import VGG


class Deepcrack(tf.keras.models):
    def __init__(self):
        self.backbone =VGG(filters=64)
