#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : data_process.py
 # Purpose :
 # Creation Date : 廿十八年四月廿七日 (週五) 十五時十一分廿四秒
 # Last Modified : 廿十八年四月廿七日 (週五) 十五時十四分廿四秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

np.save('sat.npy', sat)
np.save('mask.npy', mask)
