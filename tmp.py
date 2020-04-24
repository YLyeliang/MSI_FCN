import os
import tensorflow as tf
from model.msi_fcn import MSI_FCN
from core.visual import writeImage
from PIL import Image
import numpy as np

# for i in range(100):
#     number = tf.random.uniform([1],0,5,dtype=tf.int32)
#     print(number)
# path =tf.train.latest_checkpoint('./work_dir/msi_fcn_3/')
# n = path.split('-')[1]
# print(n)
# a='abc'
# if a=='abc':
#     print("yes")

img_path="D:\AerialGoaf\detail\image//0108.png"
image = Image.open(img_path)
image=np.array(image)
h,w,c=image.shape
patch = image[:512,w-512:,:]
img_in = tf.convert_to_tensor(patch,dtype=tf.float32)
img_in = tf.image.resize(img_in,(256,256))
img_in=img_in/255
img_in = tf.expand_dims(img_in,0)
debug=1

# MSI_FCN
model_config = {"input_scales": 4, "dcu_gr": 16, "dense_gr": 24,"filters": 64,
                     "expansion": 2, "msc_filters": [2, 2, 2, 2],"k": (7, 5, 3, 1),
                     "up_filters": 2, "num_layers": (4, 4, 4, 4), "num_classes": 2,"use_msc":True,"use_up_block":False}
work_dir = './work_dir/msi_fcn_4scales/weights/ckpt-17'
model = MSI_FCN(**model_config)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(work_dir)
pred=model(img_in)
writeImage(pred,"pred.png")