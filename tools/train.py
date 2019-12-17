import tensorflow as tf
import numpy as np
import os
from core import fit
from core import get_dataset
from model import MSI_FCN
from core.loss import WSCE
import datetime
import argparse

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train the model")
#     parser.add_argument("--img_dir",default=)

def main():
    root = '/home/yel/yel/data/Aerialgoaf/detail/'
    # root = '/home/yel/yel/data/DeepCrack-master/dataset/Deepcrack/'
    img_dir = root + 'train'
    label_dir = root + 'trainannot'
    val_dir = root + 'val'
    vallabel_dir = root + 'valannot'
    train_ds = get_dataset(img_dir, label_dir, batch_size=4)
    val_ds = get_dataset(val_dir, vallabel_dir, batch_size=4)
    model = MSI_FCN()

    lr = tf.keras.optimizers.schedules.ExponentialDecay(2e-4,10000,0.1)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

    fit(train_ds=train_ds, val_ds=val_ds, model=model, optimizer=optimizer,
        loss_func=WSCE, work_dir='../work_dir/msi_fcn', epochs=100,fine_tune=True)

#10990

if __name__ == '__main__':
    main()