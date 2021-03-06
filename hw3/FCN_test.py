#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : FCN_test.py
 # Purpose : Loading the FCN Model and Test the Data
 # Creation Date : 廿十八年四月廿六日 (週四) 廿一時廿三分三秒
 # Last Modified : 2018年04月29日 (週日) 00時48分23秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Activation
from keras.models import Model, load_model
# GPU setting\n",
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# set_session(sess)
s = time.time()
# valid sat & mask
filepath = sys.argv[2]
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

model_name = sys.argv[1]
fcn_model = load_model(model_name)
fcn_model.summary()

result = fcn_model.predict(sat, batch_size=1)
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
pred_valid_path = sys.argv[3]
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
    mpimg.imsave(os.path.join(pred_valid_path, num + '_mask.png'), mask[i])
print(time.time() - s, 'seconds')
