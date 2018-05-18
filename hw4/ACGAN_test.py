#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : ACGAN_test.py
 # Purpose :
 # Creation Date : 廿十八年五月十六日 (週三) 十七時廿一分36秒
 # Last Modified : 廿十八年五月十八日 (週五) 廿二時32分57秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#import torch related module
import torch
from torch.autograd import Variable
import scipy.misc

from ACGAN import *

if __name__ == '__main__':
    np.random.seed(69)

    n_test_imgs = 10 
    z_dim = 100
    n_classes = 2
    test_z_sample, _ = random_sample_z_c(n_test_imgs, z_dim, n_classes)
    test_z_sample = torch.cat([test_z_sample, test_z_sample], 0)
    c = 1-np.repeat(np.eye(n_classes), n_test_imgs, 0).reshape(n_test_imgs*n_classes, n_classes, 1, 1)
    test_c_sample = Variable(torch.from_numpy(c).float()).cuda()

    G = torch.load(sys.argv[1])
    G.cuda()

    #generate some images
    G.eval()
    gen_images = G(test_z_sample, test_c_sample)
    gen_images = gen_images.data.cpu().numpy()
    gen_images = gen_images.transpose((0, 2, 3, 1))
    result = np.zeros((64*n_classes,64*n_test_imgs,3)) 
    for i in range(n_classes * n_test_imgs):
        h = int(i / n_test_imgs)
        w = i % n_test_imgs
        result[(0+h*64):(64+64*h), (0+w*64):(64+64*w), :] = gen_images[i,:,:,:]
    output_path = os.path.join(sys.argv[2], 'fig3_3.jpg')
    scipy.misc.imsave(output_path,(result+1)/2)

