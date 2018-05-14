#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : GAN.py
 # Purpose : Training a GAN model
 # Creation Date : 2018年05月03日 (週四) 13時36分05秒
 # Last Modified : 廿十八年五月十四日 (週一) 十三時48分49秒
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
    batch_size = 20
    epochs = 50
    test = True
    boardX = True
    if boardX:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('runs/'+sys.argv[1])

    print('Reading the training data of face...',)
    sys.stdout.flush()

    filepath = '../data/hw4_dataset/train/'
    face_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    face_list.sort()
    n_faces = len(face_list)
    h, w, d = 64, 64, 3
    true_np = np.empty((n_faces, h, w, d), dtype='float32')

    for i, file in enumerate(face_list):
        true_np[i] = mpimg.imread(os.path.join(filepath, file))*2-1
    print("Done!")

    #Turn the np dataset to Tensor
    true_ts = torch.from_numpy(true_np.transpose((0, 3, 1, 2))).cuda()
    #target_ts = torch.ones(n_faces,1)
    true_label_ts = torch.from_numpy(0.3*np.random.rand(n_faces,1).astype('float32')+0.7)
    true_set = Data.TensorDataset(data_tensor=true_ts, target_tensor=true_label_ts)

    true_dataloader = Data.DataLoader(dataset=true_set, 
                                    batch_size=batch_size, 
                                    shuffle=True)
    del true_np

    #print('Reading the more data of face...', )
    #sys.stdout.flush()
    #filepath = '../data/hw4_dataset/test/'
    #face_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    #face_list.sort()
    #n_faces = 10
    #h, w, d = 64, 64, 3
    #test_np = np.empty((n_faces, h, w, d), dtype='float32')

    #for i, file in enumerate(face_list[0:10]):
    #    test_np[i] = mpimg.imread(os.path.join(filepath, file))*2-1
    #print("Done!")
    #test_ts = torch.from_numpy(test_np.transpose((0, 3, 1, 2))).cuda()

    G_in = 100
    G = Generator(G_in)
    D = Discriminator(3)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()

    G_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.999))

    criterion = nn.BCELoss().cuda()

    #training with 40000 face images
    G.train()
    D.train()
    test_sample = Variable(torch.randn(32, G_in, 1, 1)).cuda()
    for epoch in range(epochs):
        start_time=time.time()
        G_loss = 0
        R_loss = 0
        F_loss = 0
        for batch_idx, (b_img, b_tar) in enumerate(true_dataloader):
            #######################
            #  Update D network   #
            #######################
            # Train D with real data
            D_optimizer.zero_grad()
            true_inputs = Variable(b_img).cuda()
            true_labels = Variable(b_tar).cuda()
            D_real_labels = D(true_inputs)
            real_loss = criterion(D_real_labels, true_labels)
            real_loss.backward()

            # Train D with fake data
            D_sample = Variable(torch.randn(batch_size,G_in, 1, 1)).cuda()
            fake_inputs = G(D_sample)
            fake_labels = Variable(torch.from_numpy(0.3*np.random.rand(batch_size,1).astype('float32'))).cuda()
            D_fake_labels = D(fake_inputs)
            fake_loss = criterion(D_fake_labels, fake_labels)
            fake_loss.backward()
            if boardX:
                writer.add_scalars('Loss of Discriminator', {'Real': real_loss.data[0], 'Fake':fake_loss.data[0]},
                                    epoch*len(true_dataloader.dataset)/batch_size+batch_idx)

            D_optimizer.step()
            R_loss += real_loss.data[0]
            F_loss += fake_loss.data[0]

            #######################
            #  Update G network   #
            #######################
            # Train D with real data
            G_optimizer.zero_grad()
            G_sample = Variable(torch.randn(batch_size, G_in, 1, 1)).cuda()
            G_fake_inputs = G(G_sample)
            #fool the Discriminator
            G_fake_labels = Variable(torch.from_numpy(0.0*np.random.rand(batch_size,1).astype('float32')+1.0)).cuda()
            DG_fake_labels = D(G_fake_inputs)
            g_loss = criterion(DG_fake_labels, G_fake_labels)
            g_loss.backward()
            G_optimizer.step()
            if boardX:
                writer.add_scalar('Loss of Generator', g_loss.data[0],
                                    epoch*len(true_dataloader.dataset)/batch_size+batch_idx)


            G_loss += g_loss.data[0]
            sys.stdout.write('\rEpoch: {} [{}/{}]\tLoss_D: {:.5f}(R:{:.5f} + F:{:.5f})\tLoss_G: {:.5f}\tTime:{:.1f}'.format(
                epoch+1, (batch_idx+1) * len(b_img), len(true_dataloader.dataset),
                real_loss.data[0]+fake_loss.data[0],
                real_loss.data[0],
                fake_loss.data[0],
                g_loss.data[0],
                time.time()-start_time))
            sys.stdout.flush()

        print('===> <Average> D_loss(R/F): {:.4f}/{:.4f} G_loss: {:.4f}'.format(R_loss/len(true_dataloader.dataset),
                                                                                F_loss/len(true_dataloader.dataset),
                                                                                G_loss/len(true_dataloader.dataset)))
            
        if test:
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
            writer.add_image('test_imresult', (result+1)/2, epoch+1)

    torch.save(G, 'GAN_G_e50_.pt')
    torch.save(D, 'GAN_D_e50_.pt')
