import tensorflow as tf
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from core.eval import eval
from model.msi_fcn import MSI_FCN
from model.fcn_vgg16 import FCN_vgg16
from core.data import get_filename_list ,get_test_data
from core.visul import writeImage,show_all_branch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualization of the model, or the inference process")
    parser.add_argument("--dataset_root",default="/home/yel/yel/data/Aerialgoaf/detail/",help="Root directory of dataset")
    parser.add_argument("--test_dir", default='test')
    parser.add_argument("--testannot_dir", default='testannot')
    parser.add_argument("--work_dir",default="./work_dir/msi_fcn_gr32",help="Directory that model saved in")
    parser.add_argument("finetune",default=False,help="Whether finetune model or not")
    return parser.parse_args()

def main():
    root = '/home/yel/yel/data/Aerialgoaf/detail/'
    # root = '/home/yel/yel/data/DeepCrack-master/dataset/Deepcrack/'
    img_dir = root + 'test'
    label_dir = root + 'testannot'
    save_dir = './visualization/msi_fcn_17/'
    test_ds= get_filename_list(img_dir,label_dir)
    # MSI_FCN
    model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,
                    "filters": 64, "expansion": 2, "msc_filters": [2, 2, 2, 2],
                    "k": (7, 5, 3, 1), "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2, "use_msc": True,
                    "use_up_block": False}
    # FCN_VGG16
    # model_config = {"filters":64,"expansion":2,"num_classes":2}
    ckpt_dir ='./work_dir/msi_fcn_4scales/'
    ckpt_name = 'ckpt-17'

    model = MSI_FCN(**model_config,display_stages=True)
    # model = FCN_vgg16(**model_config)
    checkpoint = tf.train.Checkpoint(model=model)
    if ckpt_name:
        path = os.path.join(ckpt_dir,ckpt_name)
    else:
        path = tf.train.latest_checkpoint(ckpt_dir)
    status = checkpoint.restore(path)
    if path is not None:
        print("resotre model from {}".format(path))
    n=1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_file,lab_file in test_ds:
        img, lab = get_test_data(img_file, lab_file)
        filename = img_file.split('/')[-1]
        print("inference {}th image:".format(n))
        pred =model(img)
        show_all_branch(pred,save_dir,filename,rgb=True)
        n+=1
        # writeImage(pred,os.path.join(save_dir,filename),rgb=True)


if __name__ == '__main__':
    main()