import tensorflow as tf


path =tf.train.latest_checkpoint('./work_dir/msi_fcn_3/')
n = path.split('-')[1]
print(n)
a='abc'
if a=='abc':
    print("yes")