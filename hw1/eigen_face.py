#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : eigen_face.py
 # Purpose : Use PCA to analyze the eigen face
 # Creation Date : 廿十八年三月十六日 (週五) 十六時〇分53秒
 # Last Modified : 廿十八年三月十六日 (週五) 十七時44分54秒
 # Created By : SL Chung
##############################################################
import os
import sys
import numpy as np
import cv2
from sklearn.decomposition import PCA
#path = sys.argv[1]
path = "../data/hw1_dataset/"
images = np.ndarray([40, 10, 56, 46, 3])

for filename in os.listdir(path):
    subname = filename.rsplit('.') #remove datatype name
    index = subname[0].split('_')
    index = int(index[0]), int(index[1])
    images[index[0]-1, index[1]-1]=cv2.imread(path+filename) 

train_set = images[:, 0:6 , :, :, :].reshape(240, 56, 46, 3)
test_set  = images[:, 6:10, :, :, :].reshape(240, 56, 46, 3)


pca = PCA(n_components = 240)
pca.fit()


