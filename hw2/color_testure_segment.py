#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : color_testure_segment.py
 # Purpose : Color & Texture Segmentation for images
 # Creation Date : 廿十八年三月廿八日 (週三) 十五時41分十秒
 # Last Modified : 廿十八年三月廿八日 (週三) 十七時卅分32秒
 # Created By : SL Chung
##############################################################

import cv2
import numpy as np
import scipy.io as scio
from sklearn.cluster import KMeans

import randomcolor

filter = scio.loadmat("../data/hw2_dataset/Problem2/filterBank.mat")
F = filter['F']
img_z = cv2.imread("../data/hw2_dataset/Problem2/zebra.jpg")    #331*640*3
img_m = cv2.imread("../data/hw2_dataset/Problem2/mountain.jpg") #417*640*3
s_z = img_z.shape
s_m = img_m.shape
flatten_z = img_z.reshape((s_z[0]*s_z[1], s_z[2]))
flatten_m = img_m.reshape((s_m[0]*s_m[1], s_m[2]))

#Start the KMeans Algorithm
k = 10
max_iteration = 1000
kmeans = KMeans(n_clusters = k, max_iter = max_iteration)
#cluster_z_color = kmeans.fit_predict(flatten_z).reshape((s_z[0], s_z[1]))

#Generate 10 Colors
colors = np.zeros((10,3))
rand_color = randomcolor.RandomColor()
for i in range(10):
    temp = rand_color.generate()[0]
    colors[i] = np.array([int(temp[1:3], 16), int(temp[3:5], 16), int(temp[5:7], 16)])


