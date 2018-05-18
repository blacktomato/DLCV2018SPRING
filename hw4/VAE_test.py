#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : VAE_test.py
 # Purpose : Use the VAE pytorch model to produce face data
 # Creation Date : 2018年05月12日 (週六) 01時47分19秒
 # Last Modified : 廿十八年五月十八日 (週五) 廿二時卅分34秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import time
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
import pandas as pd
from sklearn.manifold import TSNE

#import torch related module
import torch
from torch.autograd import Variable
from VAE import *

if __name__ == '__main__':
    np.random.seed(69)
    print('Reading the testing data of face...', end='' )
    sys.stdout.flush()
    filepath = os.path.join(sys.argv[2], '/test/')
    face_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    face_list.sort()
    n_faces = len(face_list)
    h, w, d = 64, 64, 3
    test_np = np.empty((n_faces, h, w, d), dtype='float32')

    for i, file in enumerate(face_list):
        test_np[i] = mpimg.imread(os.path.join(filepath, file))*2-1
    print("Done!")
    test_ts = torch.from_numpy(test_np.transpose((0, 3, 1, 2))).cuda()

    vae = torch.load(sys.argv[1])
    vae.cuda()

    #reconstruct some images
    inputs = Variable(test_ts[0:10]).cuda()
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

    output_path = os.path.join(sys.argv[3], 'fig1_4.jpg')
    scipy.misc.imsave(output_path, result+1)/2)

    #loading the attribute of the test image
    filepath = os.path.join(sys.argv[2], 'test.csv')
    cl_label = pd.read_csv(filepath)['Smiling'].as_matrix().reshape(n_faces, 1)
    n_latent = 500
    selection = np.random.choice(n_faces, n_latent)
    test_tsne = torch.from_numpy(test_np.transpose((0, 3, 1, 2))[selection]).cuda()
    test_tsne = Variable(test_tsne).cuda()

    latent = torch.Tensor().cuda()
    for i in range(int(n_latent/10)):
        latent = torch.cat([latent, vae.encoder(test_tsne[0+i*10:10+i*10])[0].data])
    
    print('Performing t-SNE...', end=''  )
    sys.stdout.flush()
    latent_embedded = TSNE(n_components=2).fit_transform(latent.cpu().numpy().reshape(n_latent,-1))
    latent_label = cl_label[selection]
    print('Done')
    
    #smiling data
    fig = plt.figure()
    plt.title("tSNE_latent")
    s = np.argwhere(latent_label==1)
    plt.scatter(latent_embedded[s,0], latent_embedded[s,1], c=[1., 0, 0] )
    ns = np.argwhere(latent_label==0)
    plt.scatter(latent_embedded[ns,0], latent_embedded[ns,1], c=[0, 1, 0] )
    print('Output: t-SNE fig'  )
    sys.stdout.flush()
    output_path = os.path.join(sys.argv[3], 'fig1_5.jpg')
    plt.savefig(output_path)

    test_set = Data.TensorDataset(data_tensor=test_ts, target_tensor=test_ts)

    batch_size = 20
    dataloader = Data.DataLoader(dataset=test_set, 
                                    batch_size=batch_size, 
                                    shuffle=True)
    
    criterion = nn.MSELoss(size_average=False)
    MSE_loss = 0
    for batch_idx, (b_img, b_tar) in enumerate(dataloader):
        inputs = Variable(b_img).cuda()
        dec = vae(inputs)    
        MSE = criterion(dec, inputs) 
        MSE_loss += MSE.data[0]

    print('===> Average loss: {:.4f}'.format(MSE_loss/len(dataloader.dataset)))
