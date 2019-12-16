import tensorflow as tf
import numpy as np

def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    logits = tf.reshape(logits, (-1, num_classes))

    epsilon = tf.constant(value=1e-10)

    logits = logits + epsilon

    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))

    # should be [batch ,num_classes]
    label_flat = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

    softmax = tf.nn.softmax(logits)

    cross_entropy = -tf.reduce_sum(tf.multiply(label_flat * tf.math.log(softmax + epsilon), head), axis=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


def WSCE(logits, labels,num_classes=2):
    """
    The weighted softmax crossentropy loss.
    """
    # The weights should be calculated by statistic-based method.
    loss_weight = np.array([
        0.3,
        9.77])  # class 0~1

    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=num_classes, head=loss_weight)