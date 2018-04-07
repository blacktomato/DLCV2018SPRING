#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : bag_of_visual_word.py
 # Purpose : Recognition with Bag of Visual Word
 # Creation Date : 廿十八年四月四日 (週三) 廿二時廿九分十六秒
 # Last Modified : 廿十八年四月八日 (週日) 〇時59分55秒
 # Created By : SL Chung
##############################################################

import os
import cv2
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import randomcolor

train100_path = '../data/hw2_dataset/Problem3/train-100/'
train10_path  = '../data/hw2_dataset/Problem3/train-10/'
test100_path  = '../data/hw2_dataset/Problem3/test-100/'

train100 = {}
train10  = {}
test100  = {}

#Read in pictrues as dictionary type
print('Reading the imgs for bag of visual word')
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

def find_kp(imgs):
    mean_kp = 0
    imgs_des = {}
    for idx in imgs:
        surf = cv2.xfeatures2d.SURF_create(500)
        while(True):
            kp, des = surf.detectAndCompute(imgs[idx], None)
            if len(kp) <= 30:
                mean_kp = mean_kp + len(kp)
                imgs_des[idx] = des
                if idx == 0 and imgs == train10:
                    temp = cv2.drawKeypoints(imgs[0],kp,None,(255,0,0),4)
                    cv2.imwrite("train10_0.jpg", temp)
                    print('kp for train-10_0:', len(kp))
                break
            surf.setHessianThreshold(surf.getHessianThreshold() + 300)
    mean_kp = mean_kp / (idx+1)
    return mean_kp, imgs_des

#Choose the Train10 img and plot the interest point at most 30
print('Finding the key points for Train-10...')
mean_kp, train10_des = find_kp(train10)
print('Average Keypoint for Train-10 (50):', mean_kp, '\n')
'''
print('Finding the key points for Train-100...')
mean_kp, train100_des = find_kp(train100)
print('Average Keypoint for Train-100 (500):', mean_kp, '\n')

print('Finding the key points for Test-100...')
mean_kp, test100_des = find_kp(test100)
print('Average Keypoint for Test-100 (500):', mean_kp, '\n')
'''
#Generate 6 Colors
colors = np.zeros((6,3))
rand_color = randomcolor.RandomColor()
for i in range(6):
    temp = rand_color.generate()[0]
    colors[i] = np.array([int(temp[1:3], 16), int(temp[3:5], 16), int(temp[5:7], 16)])

#Start the KMeans Algorithm
k = 50
max_iteration = 5000
kmeans = KMeans(n_clusters = k, max_iter = max_iteration)

print("KMeans clustering for Train10 (k=50)...")
start_time = time.time()
alldes_train10 = np.zeros((0, 64))
for idx in train10_des:
    alldes_train10 = np.vstack((alldes_train10, train10_des[idx]))
train10_des_cluster = kmeans.fit_predict(alldes_train10)
print("--- %s seconds ---" % (time.time() - start_time))


c_3D_plot = np.random.randint(50, size=6)
for i in range(6):

print("Done!")
