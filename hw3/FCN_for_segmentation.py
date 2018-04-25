#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : FCN_for_segmentation.py
 # Purpose : Training a Fully Convolution Network for Image Segmentation
 # Creation Date : 廿十八年四月廿五日 (週三) 十一時廿八分34秒
 # Last Modified : 廿十八年四月廿五日 (週三) 十一時42分47秒
 # Created By : SL Chung
##############################################################

from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
import sys

data_path = sys.argv[1] 
print(data_path)
img_input = Input(shape=(512,512,3))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

model = Model(img_input, x)
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model.load_weights(data_path + weights_path, by_name=True)
