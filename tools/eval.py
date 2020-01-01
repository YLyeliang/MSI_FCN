import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from core.eval import eval
from core.data import get_filename_list
from model.msi_fcn import MSI_FCN
from model.fcn_vgg16 import FCN_vgg16
from model.fcd import FCD
import argparse


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train the model")
#     parser.add_argument("--img_dir",default=)

def main():
    root = '/home/yel/yel/data/Aerialgoaf/detail/'
    # root = '/home/yel/yel/data/DeepCrack-master/dataset/DeepCrack/'
    # root = '/home/yel/yel/data/road_crack/'
    img_dir = root + 'test'
    label_dir = root + 'testannot'
    test_ds = get_filename_list(img_dir, label_dir)

    # MSI_FCN
    model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,
                    "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
                    "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2,"use_msc":True,"use_up_block":False}

    # FCN_VGG16
    # model_config = {"filters": 64, "expansion": 2, "num_classes": 2}

    # FCD
    # model_config = {"growth_rate": 12, "td_filters": [48, 112, 192, 304, 464, 656, 896],
    #                 "up_filters": [1088, 816, 578, 384, 256], "down_layers": [4, 4, 4, 4, 4, 4],
    #                 "up_layers": [4, 4, 4, 4, 4], "num_classes": 2}

    ckpt_dir = '../work_dir/msi_fcn_4scales'

    # model = FCD(**model_config)
    model = MSI_FCN(**model_config)
    # model = FCN_vgg16(**model_config)
    for k, v in model_config.items():
        print("{}: {}".format(k, v))
    eval(test_ds, model, ckpt_dir=ckpt_dir, ckpt_name='ckpt-17')


if __name__ == '__main__':
    main()
