#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : eigen_face.py
 # Purpose : Use PCA to analyze the eigen face
 # Creation Date : 廿十八年三月十六日 (週五) 十六時〇分53秒
 # Last Modified : 2018年03月22日 (週四) 00時25分44秒
 # Created By : SL Chung
##############################################################
import os
import sys
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

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


pca = PCA()
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
    part_es_face = np.zeros((1,240))
    part_es_face[0][0:i] = es_face[0][0:i]
    recover_face = pca.inverse_transform(part_es_face).reshape((56,46,3))
    
    MSE = np.mean( np.square( recover_face - train_set[0].reshape((56,46,3)) ) )
    print("MSE: ", MSE)
    cv2.imwrite("n_"+str(i)+"_recover_face"+".jpg", recover_face)

K = [1, 3,  5]
N = [3,50,159]


for k in K:
    for n in N:
        #3-fold cross validation
        all = np.arange(6)
        np.random.shuffle(all)
        cut_t = [(all[0],all[1],all[2],all[3]),
                 (all[2],all[3],all[4],all[5]),
                 (all[0],all[1],all[4],all[5])]
        cut_v = [(all[4],all[5]),
                 (all[0],all[1]),
                 (all[2],all[3])]
        print("k= ", k , ", n= ", n)
        accuracy = [0,0,0]
        pca = PCA()
        target = np.repeat(np.arange(40),4)
        answer = np.repeat(np.arange(40),2)
        for i in range(3):
            train_set = images[:, cut_t[i], :, :, :].reshape(160, 56*46*3)
            valid_set = images[:, cut_v[i], :, :, :].reshape( 80, 56*46*3)

            pca.fit(train_set)

            es_train = pca.transform(train_set)[:, 0:n]
            es_valid = pca.transform(valid_set)[:, 0:n]
            
            
            neigh = KNeighborsClassifier(n_neighbors = k)
            neigh.fit(es_train, target)
            
            result = neigh.predict(es_valid)

            accuracy[i] = np.sum(answer == result) / 80

        print(", accuracy= ", np.mean(accuracy)*100, "%")

train_set  = images[:, 0:6, :, :, :].reshape(240, 56*46*3)
pca = PCA()
pca.fit(train_set)
target = np.repeat(np.arange(40),6)
answer = np.repeat(np.arange(40),4)
k = 1
n = 159
'''
for k in K:
    for n in N:
'''
es_train = pca.transform(train_set)[:, 0:n]
es_test  = pca.transform( test_set)[:, 0:n]


neigh = KNeighborsClassifier(n_neighbors = k)
neigh.fit(es_train, target)

result = neigh.predict(es_test)

accuracy = np.sum(answer == result) / 160
print("k = ", k, "n = ", n)
print("On the test_set accuracy= ", accuracy*100, "%")

