#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : GAN_test.py
 # Purpose : Load GAN_model and Produce the Random Faces
 # Creation Date : 2018年05月15日 (週二) 00時27分52秒
 # Last Modified : 2018年05月15日 (週二) 00時41分33秒
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
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torchvision import transforms

class Generator(nn.Module):
    def __init__(self, D_in):
        super(Generator, self).__init__()
        ndf = 64
        self.convtrans1 = nn.ConvTranspose2d(   D_in, ndf*16, (4,4))
        self.bn1 = nn.BatchNorm2d(ndf*16)
        self.convtrans2 = nn.ConvTranspose2d( ndf*16, ndf*8 , (4,4), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ndf*8)
        self.convtrans3 = nn.ConvTranspose2d( ndf*8 , ndf*4 , (4,4), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.convtrans4 = nn.ConvTranspose2d( ndf*4 , ndf*2 , (4,4), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(ndf*2)
        self.drop1 = nn.Dropout(0.5)
        self.convtrans5 = nn.ConvTranspose2d( ndf*2 ,     3, (4,4), stride=2, padding=1)
        self.drop2 = nn.Dropout(0.5)
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.convtrans1(x))) 
        x = F.leaky_relu(self.bn2(self.convtrans2(x))) 
        x = F.leaky_relu(self.bn3(self.convtrans3(x))) 
        x = F.leaky_relu(self.bn4(self.convtrans4(x))) 
        return F.tanh(self.convtrans5(x)) 

class Discriminator(nn.Module): 
    def __init__(self, D_in):
        super(Discriminator, self).__init__()
        #for 64x64 images
        ndf = 64
        self.conv1 = nn.Conv2d(D_in   ,  ndf*2, (4,4), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ndf*2)
        self.conv2 = nn.Conv2d( ndf*2 ,  ndf*4, (4,4), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ndf*4)
        self.conv3 = nn.Conv2d( ndf*4 ,  ndf*8, (4,4), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ndf*8)
        self.conv4 = nn.Conv2d( ndf*8 , ndf*16, (4,4), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(ndf*16)
        self.conv5 = nn.Conv2d( ndf*16 ,     1, (4,4), stride=1)
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x))) 
        x = F.leaky_relu(self.bn2(self.conv2(x))) 
        x = F.leaky_relu(self.bn3(self.conv3(x))) 
        x = F.leaky_relu(self.bn4(self.conv4(x))) 
        x = F.sigmoid(self.conv5(x))
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

if __name__ == '__main__':
    np.random.seed(69)

    G_in = 100
    G = torch.load(sys.argv[1])
    D = torch.load(sys.argv[2])
    G.cuda()
    D.cuda()

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

