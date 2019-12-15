import tensorflow as tf
import time
import os
from model.msi_fcn import MSI_FCN
from core.data import get_dataset
from core.loss import WSCE
from core.metrics import Metrics
import datetime

root = '/home/yel/yel/data/Aerialgoaf/detail/'
img_dir = root+'train'
label_dir = root+'trainannot'
train_ds= get_dataset(img_dir,label_dir,batch_size=5)
model = MSI_FCN()
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

tp = tf.metrics.TruePositives()
tn = tf.metrics.TrueNegatives()
fp = tf.metrics.FalsePositives()
fn = tf.metrics.FalseNegatives()
p = tf.keras.metrics.Precision()
r = tf.keras.metrics.Recall()
# auc = tf.keras.metrics.AUC()
acc = tf.keras.metrics.Accuracy()
MeanIou = tf.keras.metrics.MeanIoU(num_classes=2)

log_dir = './logs/'
summary_writer = tf.summary.create_file_writer(
    log_dir+"fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(model, input, label, loss_object, optimizer,summary_writer, step):
    with tf.GradientTape() as t:
        output = model(input, training=True)
        loss = loss_object(output, label)

    metrics={}
    y_true = tf.argmax(output, axis=-1)
    y_true = tf.reshape(y_true, (-1, 1))
    y_pred = tf.reshape(label, (-1, 1))
    tp.update_state(y_true, y_pred)
    tn.update_state(y_true, y_pred)
    fp.update_state(y_true, y_pred)
    fn.update_state(y_true, y_pred)
    p.update_state(y_true, y_pred)
    r.update_state(y_true, y_pred)
    acc.update_state(y_true, y_pred)
    MeanIou.update_state(y_true, y_pred)

    num_tp = tp.result().numpy()
    num_tn = tn.result().numpy()
    num_fp = fp.result().numpy()
    num_fn = fn.result().numpy()
    num_p = p.result().numpy()
    num_r = r.result().numpy()
    num_acc = acc.result().numpy()
    num_miou = MeanIou.result().numpy()

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

    gradients = t.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('acc', metrics['acc'], step=step)
        tf.summary.scalar('Iou(crack)', metrics['Iou(crack)'], step=step)
        tf.summary.scalar('MeanIou', metrics['MeanIou'], step=step)
    return metrics


def fit(train_ds, val_ds, epochs, ckpt_manager):
    for epoch in range(epochs):
        start = time.time()
        # Train
        for n, (input_image, target) in train_ds.enumerate():
            metrics = train_step(model,input_image,target,WSCE,optimizer,summary_writer,epoch)
            if (n + 1) % 10 == 0:
                print("Epoch: {}, step: {}, loss: {:.5f}, acc: {:.3f}, Iou(crack): {:.3f}, MeanIoU: {:.3f}".format(
                    epoch + 1, n + 1, metrics['loss'], metrics['acc'], metrics['Iou(crack)'], metrics['MeanIou']))
            # if (n + 1) % 100 == 0:
            #     for i, l in val_ds.take(1):
            #         train_step(i, l, epoch)
            #         print("Epoch: {}, step: {}, loss: {:.5f}, acc: {:.3f}, Iou(crack): {:.3f}, MeanIoU: {:.3f}".format(
            #             epoch + 1, n + 1, metrics['loss'], metrics['acc'], metrics['Iou(crack)'], metrics['MeanIou']))

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
    ckpt_manager.save()

fit(train_ds,None,150,ckpt_manager)