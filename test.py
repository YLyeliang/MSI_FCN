import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model.msi_fcn import MSI_FCN
from model.fcn_vgg16 import FCN_vgg16
from model.fcd import FCD
from core.eval import eval
from core.data import get_filename_list
import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Test the model")
    parser.add_argument("--root", default="D:/data/detail/", help="Root directory of dataset")
    parser.add_argument("--test", default='test', help='The name of directory containing testing images.')
    parser.add_argument("--testannot", default='testannot', help="the name of directory containing testing labels.")
    parser.add_argument("--ckpt", default="./work_dir/msi_fcn_4scale/ckpt-17",
                        help="ckpt path, it will load specific ckpt file if specify ckpt name")
    parser.add_argument("--model", default="msi_fcn", help="3 choices,'msi_fcn', 'fcn' or 'fcd'")
    return parser.parse_args()


def main():
    args = parse_args()

    test_dir = os.path.join(args.root, args.test)
    testannot_dir = os.path.join(args.root, args.testannot)

    test_ds = get_filename_list(test_dir, testannot_dir)

    # MSI_FCN
    if args.model == 'msi_fcn':
        model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24, "filters": 64,
                        "expansion": 2, "msc_filters": [2, 2, 2, 2], "k": (7, 5, 3, 1),
                        "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2, "use_msc": True,
                        "use_up_block": False}
        model = MSI_FCN(**model_config)

    # FCN-VGG
    elif args.model == 'fcn':
        model_config = {"filters": 64, "expansion": 2, "num_classes": 2}
        model = FCN_vgg16(**model_config)

    # FCD
    elif args.model == 'fcd':
        model_config = {"growth_rate": 12, "td_filters": [48, 112, 192, 304, 464, 656, 896],
                        "up_filters": [1088, 816, 578, 384, 256], "down_layers": [4, 4, 4, 4, 4, 4],
                        "up_layers": [4, 4, 4, 4, 4], "num_classes": 2}
        model = FCD(**model_config)
    else:
        raise ValueError("args.model should be 'msi_fcn', 'fcn' or 'fcd'.")

    ckpt = args.ckpt
    assert ckpt is not None
    # print model params
    # model.build(input_shape=(None,256,256,3))
    # print(model.summary())
    for k, v in model_config.items():
        print("{}: {}".format(k, v))
    eval(test_ds, model, ckpt_dir=ckpt)


if __name__ == '__main__':
    main()
