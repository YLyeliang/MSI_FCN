import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from core.train import fit
from core.data import get_dataset
from model.msi_fcn import MSI_FCN
from model.fcn_vgg16 import FCN_vgg16
from model.fcd import FCD
from core.loss import WSCE
import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--dataset_root",default="/home/yel/yel/data/Aerialgoaf/detail/",help="Root directory of dataset")
    parser.add_argument("--train_dir",default='train')
    parser.add_argument("--trainannot_dir",default='trainannot')
    parser.add_argument("--val_dir",default='val')
    parser.add_argument("--valannot_dir",default='valannot')
    parser.add_argument("--test_dir", default='test')
    parser.add_argument("--testannot_dir", default='testannot')
    parser.add_argument("--work_dir",default="./work_dir/msi_fcn_gr32",help="Directory that model saved in")
    parser.add_argument("finetune",default=False,help="Whether finetune model or not")
    return parser.parse_args()


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

# model msi_fcn_4:     model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}

# model msi_fcn_3scales:     model_config = {"input_scales": 3, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}

# model msi_fcn_3:     model_config = {"input_scales": 3, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}

# model msi_fcn_2scales:     model_config = {"input_scales": 2, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}

# model msi_fcn_2:     model_config = {"input_scales": 2, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}

# model msi_fcn_1scales:     model_config = {"input_scales": 1, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}

# model msi_fcn CFD:     model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,
#                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
#                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2}
#     work_dir='./work_dir/msi_fcn_CFD'

# model msi_fcn common-skip:     model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,
# #                     "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
# #                     "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2,"use_msc":Flase}
#     work_dir='./work_dir/msi_fcn_commonskip'

# model fcn_vgg16: model_config = {"filters":64,"expansion":2,"num_classes":2}
#       work_dir ='./work_dir/fcn_vgg16'

# model fcd_103:   model_config = {"growth_rate":16,"td_filters":[48,112,192,304,464,656,896],
#                  "up_filters":[1088,816,578,384,256],"down_layers":[4,5,7,10,12,15],
#                  "up_layers":[12,10,7,5,4],"num_classes":2}

# model fcd_56: model_config = {"growth_rate":12,"td_filters":[48,112,192,304,464,656,896],
# #                  "up_filters":[1088,816,578,384,256],"down_layers":[4,4,4,4,4,4],
# #                  "up_layers":[4,4,4,4,4],"num_classes":2}

# model msi_fcn_up_dense_gr_24
# model msi_fcn common-skip:       model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,"filters": 64,
#                          "expansion": 2, "msc_filters": [2, 2, 2, 2],"k": (7, 5, 3, 1),
#                          "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2,"use_msc":False,"use_up_block":True}
#     work_dir='./work_dir/msi_fcn_skip_up_block'

# model msi_fcn_skip_gr16:  model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 16,"filters": 64,
# #                          "expansion": 2, "msc_filters": [2, 2, 2, 2],"k": (7, 5, 3, 1),
# #                          "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2,"use_msc":False,"use_up_block":False}


def main():
    root = '/home/yel/yel/data/Aerialgoaf/detail/'
    # root = '/home/yel/yel/data/DeepCrack-master/dataset/DeepCrack/'
    # root = '/home/yel/yel/data/road_crack/'
    img_dir = root + 'train'
    label_dir = root + 'trainannot'
    val_dir = root + 'val'
    vallabel_dir = root + 'valannot'
    train_ds = get_dataset(img_dir, label_dir, batch_size=6)
    val_ds = get_dataset(val_dir, vallabel_dir, batch_size=6)
    # val_ds = None
    # fine_tune = True
    fine_tune=False

    # MSI_FCN
    model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 16,"filters": 64,
                         "expansion": 2, "msc_filters": [2, 2, 2, 2],"k": (7, 5, 3, 1),
                         "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2,"use_msc":False,"use_up_block":False}
    # FCN-VGG
    # model_config = {"filters": 64, "expansion": 2, "num_classes": 2}

    # FCD
    # model_config = {"growth_rate": 12, "td_filters": [48, 112, 192, 304, 464, 656, 896],
    #                 "up_filters": [1088, 816, 578, 384, 256], "down_layers": [4, 4, 4, 4, 4, 4],
    #                 "up_layers": [4, 4, 4, 4, 4], "num_classes": 2}

    work_dir = './work_dir/msi_fcn_skip_gr16'

    # model = FCD(**model_config)
    # model = FCN_vgg16(**model_config)
    model = MSI_FCN(**model_config)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(2e-4, 5000, 0.95)
    optimizer = tf.keras.optimizers.Adam(lr)
    for k, v in model_config.items():
        print("{}: {}".format(k, v))
    fit(train_ds=train_ds, val_ds=val_ds, model=model, optimizer=optimizer,
        loss_func=WSCE, work_dir=work_dir, epochs=100, fine_tune=fine_tune)


if __name__ == '__main__':
    main()
