#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : VAE.py
 # Purpose : Training a Variational AutoEncoder model
 # Creation Date : 2018年05月03日 (週四) 13時34分13秒
 # Last Modified : 廿十八年五月四日 (週五) 十七時廿六分八秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#import torch related module
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import transforms

print('Reading the training data of face...', end='')
sys.stdout.flush()
filepath = '../data/hw4_dataset/train/'
face_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
face_list.sort()
n_faces = len(face_list)
h=64
w=64
d=3
face_np = np.empty((n_faces, h, w, d), dtype='float32')
for i, file in enumerate(face_list):
    face_np[i] = mpimg.imread(os.path.join(filepath, file))/255
print("Done!")

#Turn the np dataset to Tensor
face_ts = torch.from_numpy(face_np.transpose((0, 3, 1, 2)))

