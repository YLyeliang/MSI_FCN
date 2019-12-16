import tensorflow as tf
import numpy as np


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return tf.math.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(predictions, labels):
    """
    This is for multi-classes metric calculation.
    :param predictions:
    :param labels:
    :return:
    """
    num_class = predictions.shape[3]
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist

class Metrics():
    def __init__(self):
        self.tp = tf.metrics.TruePositives()
        self.tn = tf.metrics.TrueNegatives()
        self.fp = tf.metrics.FalsePositives()
        self.fn = tf.metrics.FalseNegatives()
        self.p = tf.keras.metrics.Precision()
        self.r = tf.keras.metrics.Recall()
        # auc = tf.keras.metrics.AUC()
        self.acc = tf.keras.metrics.Accuracy()
        self.MeanIou = tf.keras.metrics.MeanIoU(num_classes=2)

    def calculate(self,true,pred):
        metrics = {}
        y_true = tf.argmax(true, axis=-1)
        y_true = tf.reshape(y_true, (-1, 1))
        y_pred = tf.reshape(pred, (-1, 1))

        self.tp.update_state(y_true, y_pred)
        self.tn.update_state(y_true, y_pred)
        self.fp.update_state(y_true, y_pred)
        self.fn.update_state(y_true, y_pred)
        self.p.update_state(y_true, y_pred)
        self.r.update_state(y_true, y_pred)
        self.acc.update_state(y_true, y_pred)
        self.MeanIou.update_state(y_true, y_pred)

        num_tp = self.tp.result().numpy()
        num_tn = self.tn.result().numpy()
        num_fp = self.fp.result().numpy()
        num_fn = self.fn.result().numpy()
        num_p = self.p.result().numpy()
        num_r = self.r.result().numpy()
        num_acc = self.acc.result().numpy()
        num_miou = self.MeanIou.result().numpy()

        metrics['tp'] = num_tp
        metrics['tn'] = num_tn
        metrics['fp'] = num_fp
        metrics['fn'] = num_fn
        metrics['p'] = num_p
        metrics['r'] = num_r
        metrics['acc'] = num_acc
        metrics['IoU(crack)'] = num_tp / (num_tp + num_fp + num_fn)
        metrics['Iou(background)'] = num_tn / (num_tn + num_fn + num_fp)
        metrics['MeanIoU'] = num_miou
        return metrics

def show_metrics(true, pred, train=True):
    """
    Show metrics. This is a binary-classes metric calculation.
    :param true:
    :param pred:
    :return:
    """
    metrics = {}
    y_pred = tf.argmax(pred, axis=-1)
    y_pred = tf.reshape(y_pred, (-1, 1))
    y_true = tf.reshape(true, (-1, 1))

    tp = tf.metrics.TruePositives()
    tn = tf.metrics.TrueNegatives()
    fp = tf.metrics.FalsePositives()
    fn = tf.metrics.FalseNegatives()
    p = tf.keras.metrics.Precision()
    r = tf.keras.metrics.Recall()
    # auc = tf.keras.metrics.AUC()
    acc = tf.keras.metrics.Accuracy()
    MeanIou = tf.keras.metrics.MeanIoU(num_classes=2)

    tp.update_state(y_true, y_pred)
    tn.update_state(y_true, y_pred)
    fp.update_state(y_true, y_pred)
    fn.update_state(y_true, y_pred)
    p.update_state(y_true, y_pred)
    r.update_state(y_true, y_pred)
    acc.update_state(y_true, y_pred)
    MeanIou.update_state(y_true, y_pred)

    num_tp = tp.result()
    num_tn = tn.result()
    num_fp = fp.result()
    num_fn = fn.result()
    num_p = p.result()
    num_r = r.result()
    num_acc = acc.result()
    num_miou= MeanIou.result()

    metrics['tp'] = num_tp
    metrics['tn'] = num_tn
    metrics['fp'] = num_fp
    metrics['fn'] = num_fn
    metrics['p'] = num_p
    metrics['r'] = num_r
    metrics['acc'] = num_acc
    metrics['IUcrack'] = num_tp / (num_tp + num_fp + num_fn)
    metrics['IUbackground'] = num_tn / (num_tn + num_fn + num_fp)
    metrics['MIU'] = num_miou
    return metrics
    # sens = tf.keras.metrics.SensitivityAtSpecificity()
    # spe = tf.keras.metrics.SpecificityAtSensitivity()



# debug
# m = tf.keras.metrics.MeanIoU(num_classes=2)
# m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
# print('Final result: ', m.result().numpy())  # Final result: 0.33

# m = tf.keras.metrics.TruePositives()
# true = tf.random.uniform([10,3,3,2],0,maxval=2,dtype=tf.int32)
# b = tf.argmax(true,axis=-1)
# debug=1
# pred = tf.random.uniform([10,1],0,maxval=2,dtype=tf.int32)
# m.update_state(true,pred)
# num = m.result()
# b=num.numpy()
# print(num)

# print('Final result: ', m.result().numpy())  # Final result: 2
