#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : eigen_face.py
 # Purpose : Use PCA to analyze the eigen face
 # Creation Date : 廿十八年三月十六日 (週五) 十六時〇分53秒
 # Last Modified : 2018年03月20日 (週二) 17時25分25秒
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

train_set = images[:, 0:6 , :, :, :].reshape(240, 56*46*3)
test_set  = images[:, 6:10, :, :, :].reshape(160, 56*46*3)


pca = PCA(whiten=True)
pca.fit(train_set)
eigenface = pca.components_
scale = np.max(eigenface, axis=1) - np.min(eigenface, axis=1)
eigenface = (eigenface - np.min(eigenface, axis=1).reshape(240, 1)) / scale.reshape(240,1) * 255

cv2.imwrite("meanface.jpg", pca.mean_.reshape((56,46,3)))
for i in range(3):
    cv2.imwrite("eigenF_"+str(i)+".jpg", eigenface[i].reshape((56,46,3)))

#transform face 1_1 to eigen space
es_face = pca.transform(train_set[0].reshape((1,56*46*3)))

#partially recover face 1_1
n = [3, 50 ,100, 239]
for i in n:
    part_es_face = es_face
    part_es_face[0][3::] = 0
    recover_face = pca.inverse_transform(part_es_face).reshape((56,46,3))
    
    cv2.imwrite("eigenF_"+str(i)+".jpg", eigenface[i].reshape((56,46,3)))
