import tensorflow as tf
import time
import os
from model.msi_fcn import MSI_FCN
from core.data import get_test_data
from core.metrics import Metrics
import datetime

def eval(test_ds,
         model=MSI_FCN(),
         ckpt_dir='./work_dir/msi_fcn/ckpt-*'
         ):
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)

    if 'ckpt' not in ckpt_dir:
        path=tf.train.latest_checkpoint(ckpt_dir)
    else:
        path=ckpt_dir
    status=checkpoint.restore(path)
    print("resotre model from {}".format(path))
    Metric = Metrics()
    n = 1
    start = time.time()
    for img_file, lab_file in test_ds:
        img, lab = get_test_data(img_file, lab_file)
        print("start inference {}th image".format(n))
        pred = model(img)
        m = Metric.update_state(lab, pred, is_train=False)
        # print("p: {}, r: {}, IUcrack: {}".format( m['tp'], m['fp'], m['fn']))
        n += 1
    metric = Metric.overall_metrics()
    for key, val in metric.items():
        print("{}: {}".format(key, val))
    end = time.time()
    print("total time: ", end -start)
