import tensorflow as tf
import time
import os
from model.msi_fcn import MSI_FCN
from core.data import get_dataset
from core.loss import WSCE
from core.metrics import show_metrics
import datetime

root = '/home/yel/yel/data/Aerialgoaf/detail/'
img_dir = root + 'train'
label_dir = root + 'trainannot'
train_ds = get_dataset(img_dir, label_dir, batch_size=3)
model = MSI_FCN()
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

log_dir = './logs/'
summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# @tf.function
def train_step(model, input, label, loss_object, optimizer, show_metrics, summary_writer, step):
    with tf.GradientTape() as t:
        output = model(input, training=True)
        loss = loss_object(output, label)

    metrics = show_metrics(label,output)
    gradients = t.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('acc', metrics['acc'], step=step)
        tf.summary.scalar('IUcrack', metrics['IUcrack'], step=step)
        tf.summary.scalar('MeanIou', metrics['MIU'], step=step)
    return metrics,loss


def fit(train_ds, val_ds, epochs, ckpt_manager):
    for epoch in range(epochs):
        start = time.time()
        # Train
        for n, (input_image, target) in train_ds.enumerate():
            metrics,loss = train_step(model, input_image, target, WSCE, optimizer, show_metrics, summary_writer, epoch)
            if (n + 1) % 10 == 0:
                print("Epoch: {}, step: {}, loss: {:.5f}, acc: {:.3f}, Iou(crack): {:.3f}, MeanIoU: {:.3f}".format(
                    epoch + 1, n + 1, loss, metrics['acc'].numpy(), metrics['IUcrack'].numpy(), metrics['MIU'].numpy()))
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


fit(train_ds, None, 150, ckpt_manager)
