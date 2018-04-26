#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : FCN_for_segmentation.py
 # Purpose : Training a Fully Convolution Network for Image Segmentation
 # Creation Date : 廿十八年四月廿五日 (週三) 十一時廿八分34秒
 # Last Modified : 廿十八年四月廿六日 (週四) 十七時五分49秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Activation, Dropout
from keras.models import Model
# GPU setting\n",
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#sat & mask
filepath = '../data/hw3_dataset/train/'
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
sat = np.empty((n_sats, h, w, d))
for i, file in enumerate(sat_list):
    sat[i] = mpimg.imread(os.path.join(filepath, file))/255

mask = np.empty((n_masks, h, w, n_classes))
cate = np.eye(n_classes)
for i, file in enumerate(mask_list):
    m = mpimg.imread(os.path.join(filepath, file))
    m = (m >= 1).astype(int)
    m = 4 * m[:, :, 0] + 2 * m[:, :, 1] + m[:, :, 2]    
    mask[i, m == 3] =cate[0]  # (Cyan:   011) Urban land 
    mask[i, m == 6] =cate[1]  # (Yellow: 110) Agriculture land 
    mask[i, m == 5] =cate[2]  # (Purple: 101) Rangeland 
    mask[i, m == 2] =cate[3]  # (Green:  010) Forest land 
    mask[i, m == 1] =cate[4]  # (Blue:   001) Water 
    mask[i, m == 7] =cate[5]  # (White:  111) Barren land 
    mask[i, m == 0] =cate[6]  # (Black:  000) Unknown
    mask[i, m == 4] =cate[6]  # (Red:    100) Unknown 

img_input = Input(shape=(512,512,3))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', trainable=False)(x)
o = x
model = Model(img_input, x)
weights_path = '../data/hw3_dataset/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model.load_weights(weights_path, by_name=True)

o = model.layers[-1].output

<<<<<<< HEAD
o = Conv2D(4096, (7, 7), activation='relu', padding='same')(o)
o = Dropout(0.5)(o)
o = Conv2D(4096, (1, 1), activation='relu', padding='same')(o)
o = Dropout(0.5)(o)
o = Conv2D( n_classes, (1 ,1), kernel_initializer='he_normal')(o)
=======
>>>>>>> 41aae11820f76bf3c43981522d1207f495a99a6e
o = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), padding='same', name='FCN_convtrans1')(o)
o = Activation('softmax')(o)
fcn_model = Model( img_input , o )

fcn_model.summary()

fcn_model.compile(optimizer='adam',
<<<<<<< HEAD
	loss='categorical_crossentropy',
	metrics=['accuracy'])
fcn_model.fit(sat, mask, epochs= 1 , batch_size= 20 )
fcn_model.save('epoch1.h5')
fcn_model.fit(sat, mask, epochs= 25, batch_size= 20 )
fcn_model.save('epoch25.h5')
fcn_model.fit(sat, mask, epochs= 50, batch_size= 20 )
fcn_model.save('epoch50.h5')
=======
>>>>>>> 41aae11820f76bf3c43981522d1207f495a99a6e
