import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
