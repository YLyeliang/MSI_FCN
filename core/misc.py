from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import os

path ='/home/yel/Downloads/CrackForest-dataset-master/groundTruth/'
outpath='/home/yel/Downloads/CrackForest-dataset-master/annot/'
bpath ='/home/yel/Downloads/CrackForest-dataset-master/boundaryannot/'
files = os.listdir(path)
for i,file in enumerate(files):
    if i==93:
        debug=1
    m = loadmat(os.path.join(path,file))
    crack = m['groundTruth'][0][0][0]
    crack = np.where(crack ==2,1,0)
    crack = crack.astype(np.uint8)
    # plt.imshow(crack)
    # plt.show()
    boundary = m['groundTruth'][0][0][1]
    img_crack = Image.fromarray(crack)
    img_boundary =Image.fromarray(boundary)
    img_crack.save(os.path.join(outpath,file[:-3]+'png'))
    img_boundary.save(os.path.join(bpath,file[:-3]+'png'))
