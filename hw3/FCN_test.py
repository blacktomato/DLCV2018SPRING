#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : FCN_test.py
 # Purpose : Loading the FCN Model and Test the Data
 # Creation Date : 廿十八年四月廿六日 (週四) 廿一時廿三分三秒
 # Last Modified : 2018年04月27日 (週五) 02時55分59秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Activation
from keras.models import Model, load_model
# GPU setting\n",
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

# valid sat & mask
filepath = sys.argv[1]
print("Start Testing: ", filepath, "...")

sat_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
mask_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
sat_list.sort()
mask_list.sort()
n_sats   = len(sat_list)
n_masks = len(mask_list)
n_classes = 7
h=512
w=512
d=3
print('Reading the SAT images...')
sat = np.empty((n_sats, h, w, d))
for i, file in enumerate(sat_list):
    sat[i] = mpimg.imread(os.path.join(filepath, file))/255

fcn_model = load_model('./128_33_128_11_e50b20.h5')
fcn_model.summary()

result = fcn_model.predict(sat, batch_size=20)
result = np.argmax(result, axis=3)

color = np.array([[0 , 1., 1.],
                  [1., 1., 0 ],
                  [1., 0 , 1.],
                  [0 , 1., 0 ],
                  [0 , 0 , 1.],
                  [1., 1., 1.],
                  [0 , 0 , 0 ],
                  ])
mask = color[result]
pred_valid_path = sys.argv[2]
for i, file in enumerate(mask_list):
    num = ''
    if (i < 10):
        num = '000'+str(i)
    elif (i < 100):
        num =  '00'+str(i)
    elif (i < 1000):
        num =   '0'+str(i)
    else:
        num =       str(i)
    mpimg.imsave(pred_valid_path + num + '_mask.png', mask[i])
