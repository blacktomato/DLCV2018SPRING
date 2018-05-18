#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : VAE.py
 # Purpose : Training a Variational AutoEncoder model
 # Creation Date : 2018年05月03日 (週四) 13時34分13秒
 # Last Modified : 廿十八年五月十八日 (週五) 十六時卅分40秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#import torch related module
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torchvision import transforms

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
        #1024 dims
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
        log_sigma = F.leaky_relu(self.bnv(self.convv(x)))
        return mean, log_sigma
   
class Decoder(nn.Module):
    def __init__(self, D_in):
        super(Decoder, self).__init__()
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
        x = F.leaky_relu(self.bn1(self.convtrans1(x))) 
        x = F.leaky_relu(self.bn2(self.convtrans2(x))) 
        x = F.leaky_relu(self.bn3(self.convtrans3(x))) 
        x = F.leaky_relu(self.bn4(self.convtrans4(x))) 
        x = F.leaky_relu(self.bn5(self.convtrans5(x))) 
        x = F.leaky_relu(self.bn6(self.convtrans6(x))) 
        return F.tanh(self.bn7(self.convtrans7(x))) 

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = h_enc[0]
        log_sigma = h_enc[1]
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma
        return mu + sigma * Variable(std_z, requires_grad=False).cuda() 

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z) 
        
def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev #Tensor has no ** operation
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

if __name__ == '__main__':
    batch_size = 20
    epochs = 100
    test = False
    boardX = False
    if boardX:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('runs/exp-1')

    print('Reading the training data of face...',end='')
    sys.stdout.flush()

    filepath = '../data/hw4_dataset/test/'
    face_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    face_list.sort()
    n_faces = len(face_list)
    h, w, d = 64, 64, 3
    train_np = np.empty((n_faces, h, w, d), dtype='float32')

    for i, file in enumerate(face_list):
        train_np[i] = mpimg.imread(os.path.join(filepath, file))*2-1
    print("Done!")

    #Turn the np dataset to Tensor
    train_ts = torch.from_numpy(train_np.transpose((0, 3, 1, 2))).cuda()
    train_set = Data.TensorDataset(data_tensor=train_ts, target_tensor=train_ts)

    dataloader = Data.DataLoader(dataset=train_set, 
                                    batch_size=batch_size, 
                                    shuffle=True)
    del train_np

    if test:
        print('Reading the testing data of face...', end='')
        sys.stdout.flush()
        filepath = '../data/hw4_dataset/train/'
        face_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
        face_list.sort()
        n_faces = 10
        h, w, d = 64, 64, 3
        test_np = np.empty((n_faces, h, w, d), dtype='float32')

        for i, file in enumerate(face_list[0:10]):
            test_np[i] = mpimg.imread(os.path.join(filepath, file))*2-1
        print("Done!")
        test_ts = torch.from_numpy(test_np.transpose((0, 3, 1, 2))).cuda()


    encoder = Encoder(3)
    decoder = Decoder(1024)
    vae = VAE(encoder, decoder)
    vae.cuda()

    criterion = nn.MSELoss()
    lambda_KL = float(sys.argv[1])

    optimizer = optim.Adam(vae.parameters(), lr=1e-4, betas=(0.5,0.999))

    #training with 40000 face images
    for epoch in range(epochs):
        start_time=time.time()
        vae.train()
        train_loss = 0
        for batch_idx, (b_img, b_tar) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = Variable(b_img).cuda()
            dec = vae(inputs)    
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            MSE = criterion(dec, inputs) 
            if boardX:
                writer.add_scalar('KLD', ll.data[0],  epoch*len(dataloader.dataset)/batch_size+batch_idx)
                writer.add_scalar('MSE', MSE.data[0], epoch*len(dataloader.dataset)/batch_size+batch_idx)
            loss = MSE + lambda_KL * ll
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            sys.stdout.write('\rEpoch: {} [{}/{}]\tLoss: {:.6f}\tMSE:{:.6f} KLD:{:.6f}\tTime:{:.1f}'.format(
                epoch+1, (batch_idx+1) * len(b_img), len(dataloader.dataset),
                loss.data[0]/len(inputs),
                MSE.data[0],
                ll.data[0],
                time.time()-start_time))
            sys.stdout.flush()

        print('===> Average loss: {:.4f}'.format(train_loss/len(dataloader.dataset)))
            
        if test:
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
            if boardX:
                writer.add_image('VAE_test_imresult', (result+1)/2, epoch+1)

            if (epoch+1) %  5 == 0 or (epoch == 0):
                mpimg.imsave('e'+str(epoch+1)+'_lKL'+sys.argv[1]+'.png', (result+1)/2)
        if (epoch+1) >= 100:
            torch.save(vae, 'e'+str(epoch+1)+'_lKL'+sys.argv[1]+'.pt')
