#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : tSNE.py
 # Purpose : plot the tSNE graphs for CNN and RNN based features
 # Creation Date : 2018年06月08日 (週五) 16時32分16秒
 # Last Modified : 2018年06月08日 (週五) 21時07分07秒
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
from sklearn.manifold import TSNE
import randomcolor

#import torch related module
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.models as models

from reader import readShortVideo, getVideoList 
from ResNet import Video2Tensor
from RNN import *

resnet50 = models.resnet50(pretrained = True)
resnet50.cuda() 
resnet50.eval()

if __name__=='__main__': 
    start_time = time.time()
    batch_size = 100
    presave = True
    TSNE = True

    video_info = getVideoList(sys.argv[3])
    video_path = sys.argv[2] 
    video_category = video_info['Video_category']
    video_name = video_info['Video_name']
    video_tag = np.array(video_info['Action_labels']).astype('float') 
    del video_info

    if presave:
        cnn_video_ts = torch.load('/data/r06942052/cnn_ts.pt')
        rnn_video_ts = torch.load('/data/r06942052/rnn_ts.pt')
    else:
        cnn_video_ts = Video2Tensor(video_path, video_category, video_name)

        video_ts, video_len = Video2Seq(video_path, video_category, video_name)
        video_set = Data.TensorDataset(video_ts, torch.Tensor(video_len).long())
        videoloader = Data.DataLoader(dataset=video_set, batch_size=batch_size) 
         
        rnn = torch.load(sys.argv[1]) 
        rnn.cuda()

        #Testing
        rnn.eval()
        h_state = None
        rnn_video_ts = torch.Tensor()
        for batch_idx, (b_feature, b_len) in enumerate(videoloader):
            seq_len = b_len.tolist()
            sort_index = np.argsort(seq_len)[::-1]
            b_feature = b_feature[sort_index.tolist(), 0:max(seq_len), :]
            pack = nn_utils.rnn.pack_padded_sequence(b_feature.cuda(),
                            np.sort(seq_len)[::-1], batch_first=True)
            result, h_s = rnn(pack, h_state)
            rnn_video_ts = torch.cat([rnn_video_ts, h_s.cpu()[0][np.argsort(sort_index)]])

            result = result.detach()
        torch.save(cnn_video_ts, '/data/r06942052/cnn_ts.pt')
        torch.save(rnn_video_ts, '/data/r06942052/rnn_ts.pt')
        
    
    if TSNE:
        sys.stdout.write('Performing t-SNE...')
        cnn_embedded = TSNE(n_components=2).fit_transform(
                        cnn_video_ts.cpu().numpy())
        rnn_embedded = TSNE(n_components=2).fit_transform(
                        rnn_video_ts.cpu().detach().numpy())
        sys.stdout.write('Done')

    #Generate n Colors (0~255)
    n_colors = 11 
    colors = np.zeros((n_colors,3))
    rand_color = randomcolor.RandomColor()
    for i in range(n_colors):
        temp = rand_color.generate()[0]
        colors[i] = np.array([int(temp[1:3], 16), int(temp[3:5], 16), int(temp[5:7], 16)])

    temp = np.repeat(colors, 64, axis=0).reshape(1, -1, 3)
    temp = np.repeat(temp, 64, axis=0)
    scipy.misc.imsave('colors.jpg', temp)
    
    fig = plt.figure(1)
    plt.title("tSNE_CNN_feature")
    for i in range(11):
        s = np.nonzero(video_tag == i)[0]
        plt.scatter(cnn_embedded[s,0], cnn_embedded[s,1], c=colors[i]/255., edgecolors='none')

    sys.stdout.write('Output: t-SNE cnn')
    plt.savefig('./tSNE_cnn.png')

    fig = plt.figure(2)
    plt.title("tSNE_RNN_feature")
    for i in range(11):
        s = np.nonzero(video_tag == i)[0]
        plt.scatter(rnn_embedded[s,0], rnn_embedded[s,1], c=colors[i]/255., edgecolors='none')
    sys.stdout.write('Output: t-SNE rnn')
    plt.savefig('./tSNE_rnn.png')
