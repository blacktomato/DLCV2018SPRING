#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : RNN_S2S_test.py
 # Purpose : Test the S2S model
 # Creation Date : 2018年06月08日 (週五) 21時11分22秒
 # Last Modified : 2018年06月09日 (週六) 00時32分31秒
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
import torch.nn.utils as nn_utils
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import torchvision.models as models

from reader import readImgSeq
from RNN_S2S import *

resnet50 = models.resnet50(pretrained = True)
resnet50.cuda() 
resnet50.eval()

if __name__=='__main__': 
    start_time = time.time()
    valid_path = sys.argv[3]
    valid_ts_l = Videos2Seqs(valid_path)
    ACC = False
    if ACC:
        valid_tag_l = readLabel('/data/r06942052/HW5_data/FullLengthVideos/labels/valid') 
    
    rnn = torch.load(sys.argv[1])
    cnn = torch.load(sys.argv[2])
    rnn.cuda()
    cnn.cuda()

    #Validation
    rnn.eval()
    cnn.eval()
    v_accuracy = 0
    valid_loss = 0
    h_state = None
    total = 0
    video_category=os.listdir(valid_path)
    video_category.sort()
    for i in range(len(valid_ts_l)):
        total += len(valid_ts_l[i])

        h = rnn(valid_ts_l[i].view(1, -1, 1000).cuda(), h_state).detach()
        h = h.cpu().view(-1, 2000).cuda()
        result = cnn(h).detach()
        result = torch.argmax(result, 1).cpu()

        if ACC:
            accuracy = torch.sum(result == valid_tag_l[i].long())
            v_accuracy += accuracy
            print('Video {}: {:.3f}%'.format(i+1, accuracy.float()/len(valid_ts_l[i])*100))
        result = result.numpy() 
        with open(os.path.join(sys.argv[4], video_category[i]+'.txt'), 'w+') as f:
            for i in range(len(result)):
                f.write(str(result[i])+'\n')

    if ACC:
        v_accuracy = v_accuracy.float() / total * 100
        print('\nAverage: {:.3f}%\n'.format(v_accuracy))
        
    print('Total testing: {:.1f} second'.format(time.time() - start_time))
