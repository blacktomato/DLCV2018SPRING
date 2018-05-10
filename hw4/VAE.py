#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : VAE.py
 # Purpose : Training a Variational AutoEncoder model
 # Creation Date : 2018年05月03日 (週四) 13時34分13秒
 # Last Modified : 廿十八年五月十日 (週四) 廿一時九分卅一秒
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
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import transforms

boardX = True
if boardX:
    from tensorboardX import SummaryWriter

class Encoder(nn.Module):
    def __init__(self, D_in):
        super(Encoder, self).__init__()
        #for 64x64 images
        ndf = 64
        self.conv1 = nn.Conv2d(D_in, ndf, (3,3), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.conv2 = nn.Conv2d( ndf   ,  ndf*2, (3,3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ndf*2)
        self.conv3 = nn.Conv2d( ndf*2 ,  ndf*4, (3,3), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.conv4 = nn.Conv2d( ndf*4 ,  ndf*8, (3,3), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(ndf*8)
        self.conv5 = nn.Conv2d( ndf*8 , ndf*16, (3,3), stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(ndf*16)
        self.conv6 = nn.Conv2d( ndf*16 ,ndf*16, (3,3), stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(ndf*16)

        self.convm = nn.Conv2d( ndf*16, ndf*16, (1,1))
        self.bnm = nn.BatchNorm2d(ndf*16)
        self.convv = nn.Conv2d( ndf*16, ndf*16, (1,1))
        self.bnv = nn.BatchNorm2d(ndf*16)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x))) 
        x = F.leaky_relu(self.bn2(self.conv2(x))) 
        x = F.leaky_relu(self.bn3(self.conv3(x))) 
        x = F.leaky_relu(self.bn4(self.conv4(x))) 
        x = F.leaky_relu(self.bn5(self.conv5(x))) 
        x = F.leaky_relu(self.bn6(self.conv6(x))) 

        mean = F.leaky_relu(self.bnm(self.convm(x)))
        log_var = F.leaky_relu(self.bnv(self.convv(x)))
        return mean, log_var
   
class Decoder(nn.Module):
    def __init__(self, D_in):
        super(Encoder, self).__init__()
        ndf = 64
        #decode 1024 dims vector
        self.convtrans1 = nn.ConvTranspose2d(   D_in, ndf*16, (1,1))
        self.bn1 = nn.BatchNorm2d(ndf*16)
        self.convtrans2 = nn.ConvTranspose2d( ndf*16, ndf*16, (2,2))
        self.bn2 = nn.BatchNorm2d(ndf*16)
        self.convtrans3 = nn.ConvTranspose2d( ndf*16, ndf*8 , (3,3))
        self.bn3 = nn.BatchNorm2d(ndf*8)
        self.convtrans4 = nn.ConvTranspose2d( ndf*8 , ndf*4 , (4,4), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(ndf*4)
        self.convtrans5 = nn.ConvTranspose2d( ndf*4 , ndf*2 , (4,4), stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(ndf*2)
        self.convtrans6 = nn.ConvTranspose2d( ndf*2 , ndf   , (4,4), stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(ndf)

        self.convtrans7 = nn.ConvTranspose2d( ndf   ,      3, (4,4), stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x))) 
        x = F.leaky_relu(self.bn2(self.conv2(x))) 
        x = F.leaky_relu(self.bn3(self.conv3(x))) 
        x = F.leaky_relu(self.bn4(self.conv4(x))) 
        x = F.leaky_relu(self.bn5(self.conv5(x))) 
        x = F.leaky_relu(self.bn6(self.conv6(x))) 
        return F.tanh(self.bn7(self.conv7(x))) 

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.linear()
        self._enc_log_sigma = torch.nn.linear()
    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_simga)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma
        
        return mu + sigma * Variable(std_z, requires_grad=False) 
    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)
        
def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev #Tensor has no ** operation
    return 0.5 * (mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

if __name__ == '__main__':
    batch_size = 200
    epochs = 1000

    print('Reading the training data of face...', end='')
    sys.stdout.flush()

    filepath = '../data/hw4_dataset/train/'
    face_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    face_list.sort()
    n_faces = len(face_list)
    h, w, d = 64, 64, 3
    face_np = np.empty((n_faces, h, w, d), dtype='float32')

    for i, file in enumerate(face_list):
        face_np[i] = mpimg.imread(os.path.join(filepath, file))
    print("Done!")

    #Turn the np dataset to Tensor
    face_ts = torch.from_numpy(face_np.transpose((0, 3, 1, 2)))
    
    #for epoch in range(epochs):

