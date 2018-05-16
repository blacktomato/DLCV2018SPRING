#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : ACGAN.py
 # Purpose : Training a Auxiliary Classifier GAN model
 # Creation Date : 2018年05月03日 (週四) 13時36分13秒
 # Last Modified : 廿十八年五月十六日 (週三) 十時十一分八秒
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
        
    def forward(self, z, c):
        # c is for the class input
        x = torch.cat([z, c], 1)
        x = F.leaky_relu(self.bn1(self.convtrans1(x))) 
        x = F.leaky_relu(self.bn2(self.convtrans2(x))) 
        x = F.leaky_relu(self.bn3(self.convtrans3(x))) 
        x = F.leaky_relu(self.bn4(self.convtrans4(x))) 
        return F.tanh(self.convtrans5(x)) 

class Discriminator(nn.Module): 
    def __init__(self, D_in, n_classes=2):
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
        self.convd = nn.Conv2d( ndf*16 ,          1, (4,4), stride=1)
        self.convc = nn.Conv2d( ndf*16 ,  n_classes, (4,4), stride=1)
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x))) 
        x = F.leaky_relu(self.bn2(self.conv2(x))) 
        x = F.leaky_relu(self.bn3(self.conv3(x))) 
        x = F.leaky_relu(self.bn4(self.conv4(x))) 
        realfake = F.sigmoid(self.convd(x))
        classes  = F.softmax(self.convc(x)).view(-1, n_classes)
        return realfake, classes

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

if __name__ == '__main__':
    batch_size = 20
    epochs = 20
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
    rf_label = 0.3*np.random.rand(n_faces,1).astype('float32')+0.7
    filepath = '../data/hw4_dataset/train.csv'
    cl_label = pd.read_csv(filepath)['Smiling'].as_matrix().astype('float32').reshape(n_face, 1)
    true_label_ts = torch.from_numpy(np.hstack(rf_label, cl_label, 1-cl_label))
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
    

    n_classes = 2
    G_in = 100 + n_classes
    G = Generator(G_in)
    D = Discriminator(3)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()

    G_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.999))

    dis_criterion = nn.BCELoss().cuda()
    aux_criterion = nn.NLLLoss().cuda()

    #training with 40000 face images
    G.train()
    D.train()
    test_z_sample = Variable(torch.randn(20, 100, 1, 1)).cuda()
    c = np.random.randint(0, n_classes, (batch_size, 1)) 
    c = np.hstack((c, 1-c)).reshape(batch_size, 2, 1, 1)
    test_c_sample = Variable(torch.from_numpy(c).float).cuda()
    for epoch in range(epochs):
        start_time=time.time()
        for batch_idx, (b_img, b_tar) in enumerate(true_dataloader):
            #######################
            #  Update D network   #
            #######################
            # Train D with real data
            D_optimizer.zero_grad()
            true_inputs = Variable(b_img).cuda()
            true_labels = Variable(b_tar[:,0]).cuda()
            true_classes = Variable(b_tar[:,1:3]).cuda()
            D_real_labels, D_real_classes = D(true_inputs)

            # Train D with fake data
            z = np.random.randn(batch_size,G_in-2, 1, 1)
            c = np.random.randint(0, n_classes, (batch_size, 1)) 
            c = np.hstack((c, 1-c)).reshape(batch_size, 2, 1, 1)
            D_z_sample = Variable(torch.from_numpy(z).float).cuda()
            D_c_sample = Variable(torch.from_numpy(c).float).cuda()
            fake_inputs = G(D_z_sample, D_c_sample)
            fake_labels = Variable(torch.from_numpy(0.3*np.random.rand(batch_size,1).astype('float32'))).cuda()
            fake_classes = Variable(b_tar[:,1]).cuda()
            D_fake_labels, D_fake_classes = D(fake_inputs)

            real_loss = dis_criterion(D_real_labels, true_labels)
            fake_loss = dis_criterion(D_fake_labels, fake_labels)
            rclas_loss = aux_criterion(D_real_classes,true_classes)
            fclas_loss = aux_criterion(D_fake_classes,fake_classes)
            (real_loss + fake_loss).backward()
            (rclas_loss + fclas_loss).backward()
            if boardX:
                writer.add_scalars('Loss of Discriminator', {'Real': real_loss.data[0], 'Fake':fake_loss.data[0]},
                                    epoch*len(true_dataloader.dataset)/batch_size+batch_idx)
                writer.add_scalars('Loss of Classifier', {'Real': rclas_loss.data[0], 'Fake':fclas_loss.data[0]},
                                    epoch*len(true_dataloader.dataset)/batch_size+batch_idx)

            D_optimizer.step()
            R_loss += real_loss.data[0]
            F_loss += fake_loss.data[0]
            ACR_loss += rclas_loss.data[0]
            ACF_loss += fclas_loss.data[0]

            #######################
            #  Update G network   #
            #######################
            # Train D with real data
            G_optimizer.zero_grad()
            z = np.random.randn(batch_size,G_in-2, 1, 1)
            c = np.random.randint(0, n_classes, (batch_size, 1)) 
            c = np.hstack((c, 1-c)).reshape(batch_size, 2, 1, 1)
            G_z_sample = Variable(torch.from_numpy(z).float).cuda()
            G_c_sample = Variable(torch.from_numpy(c).float).cuda()
            G_fake_inputs = G(G_z_sample, G_c_sample)
            #fool the Discriminator
            G_fake_labels = Variable(torch.from_numpy(0.0*np.random.rand(batch_size,1).astype('float32')+1.0)).cuda()
            DG_fake_labels, DG_fake_classes = D(G_fake_inputs)
            g_loss     = dis_criterion(DG_fake_labels, G_fake_labels)
            gclas_loss = aux_criterion(DG_fake_classes, G_fake_labels)
            g_loss.backward()
            gclas_loss.backward()
            G_optimizer.step()
            if boardX:
                writer.add_scalar('Loss of Generator', g_loss.data[0],
                                    epoch*len(true_dataloader.dataset)/batch_size+batch_idx)


            G_loss += g_loss.data[0]
            status = '\rEpoch: {} [{}/{}] '
            D_status = 'Aux_Loss_D: {:.3f} (R:{:.3f} + F:{:.3f}) '
            D_status = 'Dis_Loss_D: {:.3f} (R:{:.3f} + F:{:.3f}) ' + D_status
            G_status = 'Dis_Loss_G: {:.3f} Aux_Loss_G: {:.3f} '
            sys.stdout.write((status+D_status+G_status+'Times:{.1f}').format(
                epoch+1, (batch_idx+1) * len(b_img), len(true_dataloader.dataset),  #status
                real_loss.data[0]+fake_loss.data[0],                                #D_status
                real_loss.data[0],
                fake_loss.data[0],
                rclas_loss.data[0]+fclas_loss.data[0],
                rclas_loss.data[0],
                fclas_loss.data[0],
                g_loss.data[0],
                gclas_loss.data[0],
                time.time()-start_time))
            sys.stdout.flush()

        if test:
            #generate some images
            G.eval()
            gen_images = G(test_z_sample, test_c_sample)
            gen_images = gen_images.data.cpu().numpy()
            gen_images = gen_images.transpose((0, 2, 3, 1))
            result = np.zeros((128,640,3)) 
            for i in range(20):
                h = int(i / 10)
                w = i % 10
                result[(0+h*64):(64+64*h), (0+w*64):(64+64*w), :] = gen_images[i,:,:,:]
            writer.add_image('acgan_faces', (result+1)/2, epoch+1)

    torch.save(G, 'ACGAN_G_e20_.pt')
    torch.save(D, 'ACGAN_D_e20_.pt')
