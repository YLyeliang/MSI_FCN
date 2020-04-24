import tensorflow as tf
from PIL import Image
import numpy as np
import os

def writeImage(image,filename,size=(512,512),rgb=True):
    if len(image.shape) ==4:
        image = tf.squeeze(image,0)
        image = tf.argmax(image,axis=-1)
        # image = tf.squeeze(image,axis=-1)
    elif len(image.shape) ==3:
        image= tf.argmax(image,axis=-1)
        # image = tf.squeeze(image,axis=-1)

    if rgb:
        back=[0,0,0]
        crack=[128,0,0]
        r = np.ones_like(image)
        g =np.ones_like(image)
        b = np.ones_like(image)
        label_colors = np.array([back,crack])
        for l in range(2):
            r[image==l] = label_colors[l,0]
            g[image==l]=label_colors[l,1]
            b[image==l]=label_colors[l,2]
        rgb = np.zeros((image.shape[0],image.shape[1],3))
        rgb[:,:,0] = r/1.0
        rgb[:,:,1]=g/1.0
        rgb[:,:,2] =b/1.0
        im = Image.fromarray(np.uint8(rgb))
    else:
        back=255
        crack=0
        label_clors = np.array([back,crack])
        gray = np.ones_like(image)
        for l in range(2):
            gray[image==l] = label_clors[l]
        im = np.squeeze(gray)
        im = Image.fromarray(np.uint8(im))
    im =im.resize(size,resample=Image.BILINEAR)
    im.save(filename)

def show_all_branch(pred,save_dir,img_name,rgb=True):
    names = ['out','msc_2','msc_3','msc_4']

    for i,name in enumerate(names):
        branch_dir = os.path.join(save_dir,name)
        if not os.path.exists(branch_dir):
            os.makedirs(branch_dir)
        img_save =os.path.join(branch_dir,img_name)
        writeImage(pred[i],img_save,rgb=rgb)

def gray_to_rgb(image,size=(512,512)):
    """
    The output of deepcrack is a binary mask that black means crack, and white means background.
    To fit our experiments, the color space need to change.
    :param img:
    :return:
    """
    back = [0, 0, 0]
    crack = [128, 0, 0]
    image = image.convert("L")
    image = np.array(image)
    r = np.ones_like(image)
    g = np.ones_like(image)
    b = np.ones_like(image)
    label_colors = np.array([back, crack])
    t= 80

    r[image > t] = label_colors[1, 0]
    g[image > t] = label_colors[1, 1]
    b[image > t] = label_colors[1, 2]

    r[image <= t] = label_colors[0, 0]
    g[image <= t] = label_colors[0, 1]
    b[image <= t] = label_colors[0, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    im = im.resize(size,resample=Image.BILINEAR)
    return im

def demo_gray_to_rgb():
    path = "D:/tmp/crack/comparision/512/obstacles/deepcrack"
    out_dir = "D:/tmp/crack/comparision/512/obstacles/deepcrack/rgb"
    file =os.listdir(path)
    files=[]
    suffix = "gf"
    for i in file:
        if suffix in i:
            files.append(i)
    for img in files:
        image =Image.open(os.path.join(path,img))
        image = gray_to_rgb(image)
        image.save(os.path.join(out_dir,img))

# demo_gray_to_rgb()