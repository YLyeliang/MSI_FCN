import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from core.train import fit
from core.data import get_dataset
from model.msi_fcn import MSI_FCN
from core.loss import WSCE
import datetime
import argparse

# gpus = tf.config.experimental.list_physical_devices('GPU')
# gpu_id = 1
# if gpus:
#     # Restrict TensorFlow to only use the first GPU
#     try:
#         tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print(e)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train the model")
#     parser.add_argument("--img_dir",default=)


# model msi_fcn_2:(self,input_scales=4,
#                  dcu_gr=16,
#                  dense_gr=24,
#                  filters=64,
#                  expansion=2,
#                  msc_filters=[2,2,2,2],
#                  k=(7,5,3,1),
#                  up_filters=2,
#                  num_layers=(4,4,4,4),
#                  num_classes=2):

# model msi_fcn_3:(self,input_scales=4,
#                  dcu_gr=16,
#                  dense_gr=24,
#                  filters=64,
#                  expansion=2,
#                  msc_filters=[2,2,2,2],
#                  k=(5,3,1,1),
#                  num_layers=(4,4,4,4),
#                  num_classes=2):  metric=show_metrics bottle_dense deconv=2

# model msi_fcn_4:(self,input_scales=4,
#                  dcu_gr=16,
#                  dense_gr=24,
#                  filters=64,
#                  expansion=2,
#                  msc_filters=[2,2,2,2],
#                  k=(5,3,1,1),
#                  num_layers=(4,4,4,4),
#                  num_classes=2): bottle_dense deconv=2

# model msi_fcn_5:(input_scales=4,
#                  dcu_gr=16,
#                  dense_gr=24,
#                  filters=64,
#                  expansion=2,
#                  msc_filters=[2,2,2,2],
#                  k=(7,5,3,1),
#                  up_filters=64,
#                  num_layers=(4,4,4,4),
#                  num_classes=2)

# model msi_fcn_deepcrak:(input_scales=4,
#                     dcu_gr=16,
#                     dense_gr=24,
#                     filters=64,
#                     expansion=2,
#                     msc_filters=[2, 2, 2, 2],
#                     k=(7, 5, 3, 1),
#                     up_filters=2,
#                     num_layers=(4, 4, 4, 4),
#                     num_classes=2)
#     model_config ={"input_scales":4,"dcu_gr":16,"dense_gr":24,"filters":64,"expansion":2,"msc_filters":[2,2,2,2],
#                   "k":(7,5,3,1),"up_filters":2,"num_layers":(4,4,4,4),"num_classes":2}

# model msi_fcn_4scales:     model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}


# model msi_fcn_3scales:     model_config = {"input_scales": 3, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}

# model msi_fcn_2scales:     model_config = {"input_scales": 2, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}

# model msi_fcn_2scales:     model_config = {"input_scales": 1, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}

# model msi_fcn CFD:     model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}
#     work_dir='./work_dir/msi_fcn_CFD'


def main():

    # root = '/home/yel/yel/data/Aerialgoaf/detail/'
    # root = '/home/yel/yel/data/DeepCrack-master/dataset/DeepCrack/'
    root = '/home/yel/yel/data/road_crack/'
    img_dir = root + 'test'
    label_dir = root + 'testannot'
    val_dir = root + 'train'
    vallabel_dir = root + 'trainannot'
    train_ds = get_dataset(img_dir, label_dir, batch_size=8)
    val_ds = get_dataset(val_dir, vallabel_dir, batch_size=8)
    # val_ds = None
    fine_tune = True
    # fine_tune=False
    model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,
                    "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
                    "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}
    work_dir='./work_dir/msi_fcn_CFD'

    model = MSI_FCN(**model_config)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(2e-4, 5000, 0.95)
    optimizer = tf.keras.optimizers.Adam(lr)

    fit(train_ds=train_ds, val_ds=val_ds, model=model, optimizer=optimizer,
        loss_func=WSCE, work_dir=work_dir, epochs=200, fine_tune=fine_tune)


if __name__ == '__main__':
    main()
