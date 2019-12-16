import cv2
import os
import numpy as np
from PIL import Image

label_dir="/home/yel/yel/data/DeepCrack-master/dataset/DeepCrack/train_lab"
label_out="/home/yel/yel/data/DeepCrack-master/dataset/DeepCrack/train_annot"

def deepcrack_to_goaf(src,dst):
    files = os.listdir(src)
    for file in files:
        img_dir = os.path.join(src,file)
        dst_dir = os.path.join(dst,file)
        img = Image.open(img_dir)
        img_np = np.asarray(img)
        img_np = np.where(img_np>1,1,0)
        img = img_np.astype(np.uint8)
        img_dst = Image.fromarray(img)
        if not os.path.exists(dst):
            os.makedirs(dst)
        img_dst.save(dst_dir)
        debug=1

deepcrack_to_goaf(label_dir,label_out)