#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : VAE_test.py
 # Purpose : Use the VAE pytorch model to produce face data
 # Creation Date : 2018年05月12日 (週六) 01時47分19秒
 # Last Modified : 廿十八年五月十六日 (週三) 廿三時32分56秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc

#import torch related module
import torch
from torch.autograd import Variable
from VAE import *

if __name__ == '__main__':
    np.random.seed(69)
    print('Reading the testing data of face...', )
    sys.stdout.flush()
    filepath = '../data/hw4_dataset/test/'
    face_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    face_list.sort()
    n_faces = 10
    h, w, d = 64, 64, 3
    test_np = np.empty((n_faces, h, w, d), dtype='float32')

    for i, file in enumerate(face_list[0:10]):
        test_np[i] = mpimg.imread(os.path.join(filepath, file))*2-1
    print("Done!")
    test_ts = torch.from_numpy(test_np.transpose((0, 3, 1, 2))).cuda()


    vae = torch.load(sys.argv[1])
    vae.cuda()

    #training with 40000 face images
            
    #reconstruct some images
    inputs = Variable(test_ts).cuda()
    vae.eval()
    recon_test = vae(inputs)
    recon_test = recon_test.data.cpu().numpy()
    recon_test = recon_test.transpose((0, 2, 3, 1))
    result = np.zeros((128,640,3)) 
    for i in range(10):
        result[0:64, (0+i*64):(64+64*i), :] = test_np[i,:,:,:]
        result[64:128, (0+i*64):(64+64*i), :] = recon_test[i,:,:,:]

    scipy.misc.imsave('fig1_3.jpg',(result+1)/2)

    random_sample = Variable(torch.from_numpy(np.random.randn(32,1024,1,1))).cuda().float()
    random_mean   = Variable(torch.from_numpy(np.random.randn(32,1024,1,1))).cuda().float()*15
    random_sigma  = Variable(torch.from_numpy(np.random.randn(32,1024,1,1))).cuda().float()*5
    random_sample = random_mean + random_sigma * random_sample
    random_img = vae.decoder(random_sample)
    random_img = random_img.data.cpu().numpy()
    random_img = random_img.transpose((0, 2, 3, 1))
    result = np.zeros((256,512,3)) 
    for i in range(32):
        h = int(i / 8)
        w = i % 8
        result[(0+h*64):(64+64*h), (0+w*64):(64+64*w), :] = random_img[i,:,:,:]

    scipy.misc.imsave('fig1_4.jpg',(result+1)/2)

