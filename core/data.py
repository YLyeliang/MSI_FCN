import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

HEIGHT=256
WIDTH=256

def get_dataset(train_path,trainannot_path,batch_size):
    """
    Given the image path and label path, and return the dataset.
    :param train_path:
    :param trainannot_path:
    :param batch_size:
    :return:
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    image_ds = tf.data.Dataset.list_files(os.path.join(train_path,"*"), seed=1)
    label_ds = tf.data.Dataset.list_files(os.path.join(trainannot_path,"*"), seed=1)
    # parse_data(next(iter(image_ds)),next(iter(label_ds)))
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    # img,lab=next(iter(dataset))
    # debug_show_img_label(img,lab)
    dataset = dataset.map(parse_data, num_parallel_calls=AUTOTUNE).shuffle(buffer_size=1000).batch(batch_size).prefetch(
        buffer_size=AUTOTUNE)
    return dataset

def half_dataset(train_path,trainannot_path,bg_path,bgannot_path,batch_size):
    """
    Given the image path and label path, and return the dataset.
    :param train_path:
    :param trainannot_path:
    :param batch_size:
    :return:
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    image_ds = tf.data.Dataset.list_files(os.path.join(train_path,"*"), seed=1)
    label_ds = tf.data.Dataset.list_files(os.path.join(trainannot_path,"*"), seed=1)
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    dataset = dataset.map(parse_data, num_parallel_calls=AUTOTUNE).shuffle(buffer_size=1000).batch(batch_size).prefetch(
        buffer_size=AUTOTUNE)

    image2_ds = tf.data.Dataset.list_files(os.path.join(bg_path,"*"), seed=2)
    label2_ds = tf.data.Dataset.list_files(os.path.join(bgannot_path,"*"), seed=2)
    dataset2=tf.data.Dataset.zip((image2_ds,label2_ds))
    dataset2 = dataset2.map(parse_data, num_parallel_calls=AUTOTUNE).shuffle(buffer_size=1000).batch(batch_size).prefetch(
        buffer_size=AUTOTUNE)
    dataset.concatenate(dataset2)
    return dataset

# normalizing the images to [0, 1]

def normalize(input_image):
  input_image = input_image/255.

  return input_image

def parse_data(img_file, label_file):
    # parse image and label files
    image = tf.io.read_file(img_file)
    image = tf.image.decode_png(image)
    image = tf.image.resize(image,[HEIGHT,WIDTH])
    label = tf.io.read_file(label_file)
    # goaf data, dtype should be tf.uint16. deepcrack should be tf.uint8
    # label = tf.image.decode_png(label, dtype=tf.uint16)
    label = tf.image.decode_png(label,dtype=tf.uint8)
    label = tf.image.resize(label, [HEIGHT, WIDTH])

    # data augmentation
    image = tf.image.random_brightness(image, max_delta=0.2, seed=1)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.0, seed=1)

    number = tf.random.uniform([1],0,5,dtype=tf.int32)
    if number[0]==1:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    elif number[0]==2:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    elif number[0]==3:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    elif number[0]==4:
        image = tf.image.rot90(image)
        label = tf.image.rot90(label)
    image = normalize(image)
    return image,label


def debug_show_img_label(images,labels):
    if images.ndim==4:
        for i in range(images.shape[0]):
            image = images[i]
            image =tf.squeeze(image)
            label = labels[i]
            label = tf.squeeze(label)
            plt.imshow(image,cmap='gray')
            plt.show()
            plt.imshow(label)
            plt.show()


def get_filename_list(img_path,label_path):
    files = os.listdir(img_path)

    image_filenames = []
    label_filenames = []
    for i in files:
        image_filenames.append(os.path.join(img_path,i))
        label_filenames.append(os.path.join(label_path,i))
    return zip(image_filenames, label_filenames)

# debug
# root = 'D:\\data\\detail\\'
# dataset = get_dataset(root+"train",root+"trainannot",5)
# img,lab=next(iter(dataset))
# debug_show_img_label(img,lab)

# print()

def read_img(img_file,size=(HEIGHT,WIDTH)):
    image = tf.io.read_file(img_file)
    image = tf.image.decode_png(image)
    image = tf.image.resize(image, size)
    image = normalize(image)
    image = tf.expand_dims(image,0)
    return image

def get_test_data(img_file,lab_file):
    img = read_img(img_file)
    label = tf.io.read_file(lab_file)
    label = tf.image.decode_png(label, dtype=tf.uint8)
    label = tf.image.resize(label, [HEIGHT, WIDTH])
    lab = tf.expand_dims(label,0)
    return img,lab


# root = '/home/yel/yel/data/DeepCrack-master/dataset/DeepCrack/'
# img_dir = root + 'train_img'
# label_dir = root + 'train_annot'
#
# get_dataset(img_dir,label_dir,1)