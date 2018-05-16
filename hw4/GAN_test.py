#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : GAN_test.py
 # Purpose : Load GAN_model and Produce the Random Faces
 # Creation Date : 2018年05月15日 (週二) 00時27分52秒
 # Last Modified : 廿十八年五月十六日 (週三) 廿時55分十六秒
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

from GAN import *

if __name__ == '__main__':
    np.random.seed(69)

    G_in = 100
    G = torch.load(sys.argv[1])
    G.cuda()

    test_sample = Variable(torch.from_numpy(np.random.randn(32, G_in, 1, 1))).float().cuda()
    #generate some images
    G.eval()
    gen_images = G(test_sample)
    gen_images = gen_images.data.cpu().numpy()
    gen_images = gen_images.transpose((0, 2, 3, 1))
    result = np.zeros((256,512,3)) 
    for i in range(32):
        h = int(i / 8)
        w = i % 8
        result[(0+h*64):(64+64*h), (0+w*64):(64+64*w), :] = gen_images[i,:,:,:]
    scipy.misc.imsave('fig2_3.jpg',(result+1)/2)


