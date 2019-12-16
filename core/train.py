import tensorflow as tf
import time
import os
from model import MSI_FCN
from core import get_dataset
from core import WSCE
from core import show_metrics
import datetime

@tf.function
def train_step(input, label, model, loss_func, optimizer):
    with tf.GradientTape() as t:
        output = model(input, training=True)
        loss = loss_func(output, label)

    gradients = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return output, loss


def write_sumamry(summary_writer, loss, metrics, step):
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('acc', metrics['acc'], step=step)
        tf.summary.scalar('IUcrack', metrics['IUcrack'], step=step)
        tf.summary.scalar('MeanIou', metrics['MIU'], step=step)


def fit(train_ds,
        val_ds,
        model=MSI_FCN(),
        loss_func=WSCE,
        optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        metric_func=show_metrics,
        work_dir='./work_dir/msi_fcn',
        epochs=100,
        fine_tune=False):

    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, work_dir, max_to_keep=5)
    summary_writer = tf.summary.create_file_writer(
        work_dir)
    n = 4120
    if fine_tune:
        path =tf.train.latest_checkpoint(work_dir)
        checkpoint.restore(path)
    for epoch in range(epochs):
        start = time.time()
        # Train
        for inputs, label in train_ds:
            output, loss = train_step(inputs, label, model, loss_func, optimizer)
            metrics = metric_func(label, output)
            write_sumamry(summary_writer, loss, metrics, step=n)
            if (n + 1) % 10 == 0:
                print("Epoch: {}, step: {}, loss: {:.5f}, acc: {:.3f}, Iou(crack): {:.3f}, MeanIoU: {:.3f}".format(
                    epoch + 1, n + 1, loss, metrics['acc'].numpy(), metrics['IUcrack'].numpy(), metrics['MIU'].numpy()))

            if (n + 1) % 100 == 0:
                if val_ds is not None:
                    for val_inputs, val_label in val_ds.take(1):
                        output, loss = train_step(val_inputs, val_label, model, loss_func, optimizer)
                        metrics = metric_func(val_label,output)
                        write_sumamry(summary_writer, loss, metrics, step=n)
                        print(
                            "Epoch: {}, step: {}, val_loss: {:.5f}, val_acc: {:.3f}, val_Iou(crack): {:.3f}, val_MeanIoU: {:.3f}".format(
                                epoch + 1, n + 1, loss, metrics['acc'].numpy(), metrics['IUcrack'].numpy(),
                                metrics['MIU'].numpy()))
            n+=1

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
    ckpt_manager.save()
