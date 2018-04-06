#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : bag_of_visual_word.py
 # Purpose : Recognition with Bag of Visual Word
 # Creation Date : 廿十八年四月四日 (週三) 廿二時廿九分十六秒
 # Last Modified : 廿十八年四月六日 (週五) 十七時〇分51秒
 # Created By : SL Chung
##############################################################

import os
import cv2
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

train100_path = '../data/hw2_dataset/Problem3/train-100/'
train10_path  = '../data/hw2_dataset/Problem3/train-10/'
test100_path  = '../data/hw2_dataset/Problem3/test-100/'

train100 = {}
train10  = {}
test100  = {}

#Read in pictrues as dictionary type
for cidx, category in enumerate(os.listdir(train100_path)):
    temp_path = train100_path + category + '/'
    for fidx, file in enumerate(os.listdir(temp_path)):
        train100[cidx*100+fidx] = cv2.imread(temp_path+file)
        
for cidx, category in enumerate(os.listdir(train10_path)):
    temp_path = train10_path + category + '/'
    for fidx, file in enumerate(os.listdir(temp_path)):
        train10[cidx*10+fidx] = cv2.imread(temp_path+file)

for cidx, category in enumerate(os.listdir(test100_path)):
    temp_path = test100_path + category + '/'
    for fidx, file in enumerate(os.listdir(temp_path)):
        test100[cidx*100+fidx] = cv2.imread(temp_path+file)
#Choose the Train10 img and plot the interest point at most 30
for idx in train10:
    surf = cv2.xfeatures2d.SURF_create(500)
    while(True):
        kp, des = surf.detectAndCompute(train10[idx], None)
        if len(kp) <= 30:
            print(idx, surf.getHessianThreshold(), len(kp))
            break
        surf.setHessianThreshold(surf.getHessianThreshold() + 300)
temp = cv2.drawKeypoints(train10[0],kp,None,(255,0,0),4)
cv2.imwrite("temp.jpg", temp)
