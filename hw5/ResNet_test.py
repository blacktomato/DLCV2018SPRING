#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : ResNet_test.py
 # Purpose : Test the Feature and FC for classification
 # Creation Date : 2018年05月30日 (週三) 15時55分47秒
 # Last Modified : 2018年06月02日 (週六) 15時39分15秒
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
import torch.utils.data as Data
import torchvision
import torchvision.models as models

from reader import readShortVideo, getVideoList 
from ResNet import FC_for_C, Video2Tensor

resnet50 = models.resnet50(pretrained = True)
resnet50.cuda() 
resnet50.eval()

if __name__=='__main__': 
    start_time = time.time()
    n_classes = 11
    batch_size = 100

    video_info = getVideoList(sys.argv[3])
    video_path = sys.argv[2] 
    video_category = video_info['Video_category']
    video_name = video_info['Video_name']
    video_tag = np.array(video_info['Action_labels']).astype('float') 
    video_tag = torch.from_numpy(video_tag)
    del video_info

    #Using ResNet50 in the function _ Video2Tensor
    video_ts = Video2Tensor(video_path, video_category, video_name)

    video_set = Data.TensorDataset(video_ts, video_tag.long())
    videoloader = Data.DataLoader(dataset=video_set, batch_size=batch_size) 

    Net = torch.load(sys.argv[1])
    Net.cuda()

    Net.eval()
    v_accuracy = 0
    
    with open(os.path.join(sys.argv[4], 'p1_valid.txt'), 'w+') as f:
        for batch_idx, (b_feature, b_tag) in enumerate(videoloader):
            result = torch.argmax(Net(b_feature.cuda()).detach(), 1).cpu().numpy()
            for i in range(len(result)):
                f.write(str(result[i])+'\n')
            v_accuracy += torch.sum(result == b_tag)

    v_accuracy = v_accuracy.float() / len(videoloader.dataset) * 100
    print('Accuracy: ', v_accuracy.item())
            
    print('Total time:{:.1f}'.format(time.time() - start_time))
