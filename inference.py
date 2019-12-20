import tensorflow as tf
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from core.eval import eval
from model.msi_fcn import MSI_FCN
from core.data import get_filename_list ,get_test_data
from core.visul import writeImage,show_all_branch

def main():
    root = '/home/yel/yel/data/Aerialgoaf/detail/'
    # root = '/home/yel/yel/data/DeepCrack-master/dataset/Deepcrack/'
    img_dir = root + 'test'
    label_dir = root + 'testannot'
    save_dir = './visualization/'
    test_ds= get_filename_list(img_dir,label_dir)
    ckpt_dir = './work_dir/msi_fcn_2'
    model = MSI_FCN(input_scales=4,
                    dcu_gr=16,
                    dense_gr=24,
                    filters=64,
                    expansion=2,
                    msc_filters=[2, 2, 2, 2],
                    k=(7, 5, 3, 1),
                    up_filters=2,
                    num_layers=(4, 4, 4, 4),
                    num_classes=2,
                    display_stages=True)
    checkpoint = tf.train.Checkpoint(model=model)
    path = tf.train.latest_checkpoint(ckpt_dir)
    status = checkpoint.restore(path)
    if path is not None:
        print("resotre model from {}".format(path))
    for img_file,lab_file in test_ds:
        img, lab = get_test_data(img_file, lab_file)
        filename = img_file.split('/')[-1]
        pred =model(img)
        show_all_branch(pred,save_dir,filename)
        # writeImage(pred,save_dir+filename,rgb=False)


if __name__ == '__main__':
    main()