#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : color_testure_segment.py
 # Purpose : Color & Texture Segmentation for images
 # Creation Date : 廿十八年三月廿八日 (週三) 十五時41分十秒
 # Last Modified : 廿十八年四月四日 (週三) 廿二時十八分一秒
 # Created By : SL Chung
##############################################################

import cv2
import time
import numpy as np
import scipy.io as scio
from scipy import signal 
from sklearn.cluster import KMeans

import randomcolor

filter = scio.loadmat("../data/hw2_dataset/Problem2/filterBank.mat")
F = filter['F']
img_z = cv2.imread("../data/hw2_dataset/Problem2/zebra.jpg")    #331*640*3
img_m = cv2.imread("../data/hw2_dataset/Problem2/mountain.jpg") #417*640*3
s_z = img_z.shape
s_m = img_m.shape

#Generate 10 Colors
colors = np.zeros((10,3))
rand_color = randomcolor.RandomColor()
for i in range(10):
    temp = rand_color.generate()[0]
    colors[i] = np.array([int(temp[1:3], 16), int(temp[3:5], 16), int(temp[5:7], 16)])

#Start the KMeans Algorithm
k = 10
max_iteration = 1000
kmeans = KMeans(n_clusters = k, max_iter = max_iteration)

#Which task should be done
Color_Cluster = False
Texture_Cluster = False
Color_Texture_Cluster = True

if Color_Cluster:
    flatten_z = img_z.reshape((s_z[0]*s_z[1], s_z[2]))
    flatten_m = img_m.reshape((s_m[0]*s_m[1], s_m[2]))
    print("KMeans clustering...")
    start_time = time.time()
    cluster_z_color = kmeans.fit_predict(flatten_z).reshape((s_z[0], s_z[1]))
    cluster_m_color = kmeans.fit_predict(flatten_m).reshape((s_m[0], s_m[1]))
    print("--- %s seconds ---" % (time.time() - start_time))
    #Output images
    print("Saving images...")
    img_Ccluster_z = colors[cluster_z_color]
    cv2.imwrite("Ccluster_zebra.jpg", img_Ccluster_z)
    img_Ccluster_m = colors[cluster_m_color]
    cv2.imwrite("Ccluster_mountain.jpg", img_Ccluster_m)
    print("Done!")

#Start the KMeans Algorithm
k = 6
kmeans = KMeans(n_clusters = k, max_iter = max_iteration)

if Texture_Cluster:
    #Symmetric Padding
    gray_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2GRAY)
    gray_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)
    #Use filter to transform the image shape to h*w*38
    filtered_z = np.zeros((s_z[0], s_z[1], F.shape[2]))
    filtered_m = np.zeros((s_m[0], s_m[1], F.shape[2]))
    print("Using the filter to find texture...")
    start_time = time.time()
    for i in range(F.shape[2]):
        filtered_z[:,:,i] = signal.correlate2d(gray_z, F[:,:,i], boundary='symm', mode='same')
        filtered_m[:,:,i] = signal.correlate2d(gray_m, F[:,:,i], boundary='symm', mode='same')
    print("--- %s seconds ---" % (time.time() - start_time))
    filtered_z = filtered_z.reshape((s_z[0]*s_z[1], F.shape[2]))
    filtered_m = filtered_m.reshape((s_m[0]*s_m[1], F.shape[2]))

    print("KMeans clustering...")
    start_time = time.time()
    cluster_z_texture = kmeans.fit_predict(filtered_z).reshape((s_z[0], s_z[1]))
    cluster_m_texture = kmeans.fit_predict(filtered_m).reshape((s_m[0], s_m[1]))
    print("--- %s seconds ---" % (time.time() - start_time))

    #Output images
    print("Saving images...")
    start_time = time.time()
    img_Tcluster_z = colors[cluster_z_texture]
    cv2.imwrite("Tcluster_zebra.jpg", img_Tcluster_z)
    img_Tcluster_m = colors[cluster_m_texture]
    cv2.imwrite("Tcluster_mountain.jpg", img_Tcluster_m)
    print("Done!")

#Start the KMeans Algorithm
k = 6
kmeans = KMeans(n_clusters = k, max_iter = max_iteration)

if Color_Texture_Cluster:
    flatten_z = img_z.reshape((s_z[0]*s_z[1], s_z[2]))
    flatten_m = img_m.reshape((s_m[0]*s_m[1], s_m[2]))

    #Symmetric Padding
    gray_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2GRAY)
    gray_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)
    #Use filter to transform the image shape to h*w*38
    filtered_z = np.zeros((s_z[0], s_z[1], F.shape[2]))
    filtered_m = np.zeros((s_m[0], s_m[1], F.shape[2]))
    print("Using the filter to find texture...")
    start_time = time.time()
    for i in range(F.shape[2]):
        filtered_z[:,:,i] = signal.correlate2d(gray_z, F[:,:,i], boundary='symm', mode='same')
        filtered_m[:,:,i] = signal.correlate2d(gray_m, F[:,:,i], boundary='symm', mode='same')
    print("--- %s seconds ---" % (time.time() - start_time))
    filtered_z = filtered_z.reshape((s_z[0]*s_z[1], F.shape[2]))
    filtered_m = filtered_m.reshape((s_m[0]*s_m[1], F.shape[2]))
    
    print("Concatenate texture and color dimension...")
    CT_z = np.concatenate( (flatten_z, filtered_z), 1)
    CT_m = np.concatenate( (flatten_m, filtered_m), 1)

    print("KMeans clustering...")
    start_time = time.time()
    cluster_z_ct = kmeans.fit_predict(CT_z).reshape((s_z[0], s_z[1]))
    cluster_m_ct = kmeans.fit_predict(CT_m).reshape((s_m[0], s_m[1]))
    print("--- %s seconds ---" % (time.time() - start_time))

    #Output images
    print("Saving images...")
    start_time = time.time()
    img_CTcluster_z = colors[cluster_z_ct]
    cv2.imwrite("CTcluster_zebra.jpg", img_CTcluster_z)
    img_CTcluster_m = colors[cluster_m_ct]
    cv2.imwrite("CTcluster_mountain.jpg", img_CTcluster_m)
    print("Done!")
