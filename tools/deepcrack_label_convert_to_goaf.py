import cv2
import os
import numpy as np
from PIL import Image

label_dir="/home/yel/yel/data/road_crack/testannot"
label_out="/home/yel/yel/data/road_crack/deepformat/test_lab"

def deepcrack_to_goaf(src,dst):
    files = os.listdir(src)
    for i,file in enumerate(files):
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

def goaf_to_deepcrack(src,dst):
    files = os.listdir(src)
    for i,file in enumerate(files):

        img_dir = os.path.join(src,file)
        dst_dir = os.path.join(dst,file)
        img = Image.open(img_dir)
        img_np = np.asarray(img)
        img_np = np.where(img_np>0,255,0)
        img = img_np.astype(np.uint8)
        img_dst = Image.fromarray(img)
        if not os.path.exists(dst):
            os.makedirs(dst)
        img_dst.save(dst_dir)

# deepcrack_to_goaf(label_dir,label_out)
goaf_to_deepcrack(label_dir,label_out)