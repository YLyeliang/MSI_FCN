import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from core.train import fit
from core.data import get_dataset,half_dataset
from model.msi_fcn import MSI_FCN
from model.fcn_vgg16 import FCN_vgg16
from model.fcd import FCD
from core.loss import WSCE
import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--root",default="D:/data/detail/",help="Root directory of dataset")
    parser.add_argument("--train",default='train',help='The name of directory containing training images.')
    parser.add_argument("--trainannot",default='trainannot',help="the name of directory containing training labels.")
    parser.add_argument("--val",default='val')
    parser.add_argument("--valannot",default='valannot')
    parser.add_argument("--work_dir",default="./work_dir/msi_fcn_4scales",help="Directory that model saved in")
    parser.add_argument("--model",default="msi_fcn",help="3 choices,'msi_fcn', 'fcn' or 'fcd'")
    parser.add_argument("--lr",default=2e-4,help="initial learning rate")
    parser.add_argument("--finetune",default=False,help="Whether finetune model or not")
    return parser.parse_args()

def main():
    args=parse_args()

    train_dir=os.path.join(args.root,args.train)
    trainannot_dir = os.path.join(args.root,args.trainannot)
    val_dir = os.path.join(args.root,args.val)
    valannot_dir = os.path.join(args.root,args.valannot)

    train_ds = get_dataset(train_dir, trainannot_dir, batch_size=4)
    val_ds = get_dataset(val_dir, valannot_dir, batch_size=4)
    # val_ds = None
    fine_tune = args.finetune
    # fine_tune=False

    # MSI_FCN
    if args.model=='msi_fcn':
        model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,"filters": 64,
                         "expansion": 2, "msc_filters": [2, 2, 2, 2],"k": (7, 5, 3, 1),
                         "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2,"use_msc":True,"use_up_block":False}
        model = MSI_FCN(**model_config)

    # FCN-VGG
    elif args.model=='fcn':
        model_config = {"filters": 64, "expansion": 2, "num_classes": 2}
        model = FCN_vgg16(**model_config)

    # FCD
    elif args.model=='fcd':
        model_config = {"growth_rate": 12, "td_filters": [48, 112, 192, 304, 464, 656, 896],
                    "up_filters": [1088, 816, 578, 384, 256], "down_layers": [4, 4, 4, 4, 4, 4],
                    "up_layers": [4, 4, 4, 4, 4], "num_classes": 2}
        model = FCD(**model_config)
    else:
        raise ValueError("args.model should be 'msi_fcn', 'fcn' or 'fcd'.")

    work_dir = args.work_dir
    # print model params
    # model.build(input_shape=(None,256,256,3))
    # print(model.summary())
    lr = tf.keras.optimizers.schedules.ExponentialDecay(2e-4, 5000, 0.95)
    optimizer = tf.keras.optimizers.Adam(lr)
    for k, v in model_config.items():
        print("{}: {}".format(k, v))
    fit(train_ds=train_ds, val_ds=val_ds, model=model, optimizer=optimizer,
        loss_func=WSCE, work_dir=work_dir, epochs=60, fine_tune=fine_tune)


if __name__ == '__main__':
    main()
