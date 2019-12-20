import tensorflow as tf
import numpy as np
import os
from core.eval import eval
from core.data import get_filename_list
from model.msi_fcn import MSI_FCN
import argparse

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train the model")
#     parser.add_argument("--img_dir",default=)

def main():
    root = '/home/yel/yel/data/Aerialgoaf/detail/'
    # root = '/home/yel/yel/data/DeepCrack-master/dataset/Deepcrack/'
    img_dir = root + 'test'
    label_dir = root + 'testannot'
    test_ds= get_filename_list(img_dir,label_dir)

    model = MSI_FCN(input_scales=4,
                    dcu_gr=16,
                    dense_gr=24,
                    filters=64,
                    expansion=2,
                    msc_filters=[2, 2, 2, 2],
                    k=(7, 5, 3, 1),
                    up_filters=2,
                    num_layers=(4, 4, 4, 4),
                    num_classes=2)
    eval(test_ds,model,ckpt_dir='../work_dir/msi_fcn_2')


#10990

if __name__ == '__main__':
    main()