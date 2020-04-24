import os

import tensorflow as tf
import numpy as np
from PIL import Image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return tf.math.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(predictions, labels):
    """
    This is for multi-classes metric calculation by calculating confusion matrix.
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
        self.metrics = {}

        # accumulate
        self.metrics['tp'] = []
        self.metrics['tn'] =[]
        self.metrics['fp'] =[]
        self.metrics['fn'] =[]

        self.tp = tf.metrics.TruePositives()
        self.tn = tf.metrics.TrueNegatives()
        self.fp = tf.metrics.FalsePositives()
        self.fn = tf.metrics.FalseNegatives()
        self.p = tf.keras.metrics.Precision()
        self.r = tf.keras.metrics.Recall()
        auc = tf.keras.metrics.AUC()
        self.acc = tf.keras.metrics.Accuracy()
        self.MeanIou = tf.keras.metrics.MeanIoU(num_classes=2)

    def update_state(self,true,pred,is_train=True):

        if is_train:
            y_pred = tf.argmax(pred, axis=-1)
            y_pred = tf.reshape(y_pred, [-1])
            y_true = tf.reshape(true, [-1])
        else:
            y_pred = tf.argmax(pred, axis=-1)
            y_pred = tf.reshape(y_pred,[-1])
            y_true = tf.reshape(true,[-1])

        self.tp.reset_states()
        self.tn.reset_states()
        self.fp.reset_states()
        self.fn.reset_states()
        self.p.reset_states()
        self.r.reset_states()
        self.acc.reset_states()
        self.MeanIou.reset_states()

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

        if is_train:
            num_p = self.p.result().numpy()
            num_r = self.r.result().numpy()
            num_acc = self.acc.result().numpy()
            num_miou = self.MeanIou.result().numpy()

            self.metrics['p'] = num_p
            self.metrics['r'] = num_r
            self.metrics['acc'] = num_acc
            self.metrics['IUcrack'] = num_tp / (num_tp + num_fp + num_fn)
            self.metrics['IUbackground'] = num_tn / (num_tn + num_fn + num_fp)
            self.metrics['MIU'] = num_miou

        self.metrics['tp'].append(num_tp)
        self.metrics['tn'].append(num_tn)
        self.metrics['fp'].append(num_fp)
        self.metrics['fn'].append(num_fn)

        return self.metrics

    def overall_metrics(self):
        tp_all = np.sum(self.metrics['tp'])
        tn_all = np.sum(self.metrics['tn'])
        fp_all = np.sum(self.metrics['fp'])
        fn_all = np.sum(self.metrics['fn'])

        overall_metrics={}
        acc = (tp_all+tn_all)/(tp_all+tn_all+fp_all+fn_all)
        specificity = tn_all/(tn_all+fp_all)
        precision = tp_all/(tp_all+fp_all)
        recall = tp_all/(tp_all+fn_all)
        gmean = np.sqrt(recall*specificity)
        f1 = 2*recall*precision/(recall+precision)
        IUcrack = tp_all/(tp_all+fp_all+fn_all)
        IUback = tn_all/(tn_all+fn_all+fp_all)
        MIU = (IUcrack+IUback)/2
        bacc = (recall+specificity)/2

        overall_metrics['accuracy'] = acc
        overall_metrics['Balanced accuracy']= bacc
        overall_metrics['precision']=precision
        overall_metrics['recall']=recall
        overall_metrics['gmean']=gmean
        overall_metrics['f1']=f1
        overall_metrics['IUcrack']=IUcrack
        overall_metrics['IUback']=IUback
        overall_metrics['MIU']=MIU

        return overall_metrics

def metrics_with_images(pred_dir,lab_dir):
    files = os.listdir(pred_dir)
    metrics = Metrics()
    for file in files:
        pred_file = os.path.join(pred_dir,file)

        lab_file = os.path.join(lab_dir,file[:-7]+".png")
        pred,true = decode_image_label(pred_file,lab_file)
        metrics.update_state(true,pred,is_train=False)
    metric = metrics.overall_metrics()
    for k,v in metric.items():
        print("{}: {}".format(k,v))

def decode_image_label(pred_file,lab_file):
    """ RGB to Gray and calculate the metrics"""
    image = tf.io.read_file(pred_file)
    image = tf.image.decode_png(image)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.where(image>0,1.,0.)
    image = tf.image.resize(image,(256,256))

    label = tf.io.read_file(lab_file)
    label = tf.image.decode_png(label, dtype=tf.uint8)
    label = tf.image.resize(label,(256,256))
    return image,label

# pred_format = "/home/yel/yel/experiments/deepcrack_{}"
# true_format = "/home/yel/yel/experiments/condition{}"
# for i in range(1,4):
#     pred_dir = pred_format.format(i)
#     true_dir = true_format.format(i)
#     print("condition: {}".format(i))
#     metrics_with_images(pred_dir,true_dir)
#     print()




# debug
# m = tf.keras.metrics.MeanIoU(num_classes=2)
# m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
# print('Final result: ', m.result().numpy())  # Final result: 0.33

# m = tf.keras.metrics.TruePositives()
# true = tf.random.uniform([10,256,256,1],0,maxval=2,dtype=tf.int32)
# b = tf.argmax(true,axis=-1)
# debug=1
# pred = tf.random.uniform([10,256,256,2],0,maxval=2,dtype=tf.int32)
# m.update_state(true,pred)
# num = m.result()
# b=num.numpy()
# print(num)
# import time
# s =time.time()
# Metric =Metrics()
# metric=Metric.update_state(true,pred,True)
# print(metric)
# ovr_metric = Metric.overall_metrics()
# # print(ovr_metric)
# e =time.time()
# print(e-s)

# print('Final result: ', m.result().numpy())  # Final result: 2
