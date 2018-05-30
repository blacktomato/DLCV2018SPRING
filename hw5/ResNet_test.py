#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : ResNet_test.py
 # Purpose : Test the Feature and FC for classification
 # Creation Date : 2018年05月30日 (週三) 15時55分47秒
 # Last Modified : 2018年05月30日 (週三) 17時02分30秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import pandas as pd
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

    valid_info = getVideoList(sys.argv[3])
    valid_path = sys.argv[2] 
    valid_category = valid_info['Video_category']
    valid_name = valid_info['Video_name']
    valid_tag = np.array(valid_info['Action_labels']).astype('float') 
    valid_tag = torch.from_numpy(valid_tag)
    del valid_info

    #Using ResNet50 in the function _ Video2Tensor
    valid_ts = Video2Tensor(valid_path, valid_category, valid_name)

    valid_set = Data.TensorDataset(valid_ts, valid_tag.long())
    validloader = Data.DataLoader(dataset=valid_set, batch_size=batch_size) 

    Net = torch.load(sys.argv[1])
    Net.cuda()

    Net.eval()
    v_accuracy = 0
    
    with open(os.path.join(sys.argv[4], 'p1_valid.txt'), 'w+') as f:
        for batch_idx, (b_feature, b_tag) in enumerate(validloader):
            result = torch.argmax(Net(b_feature.cuda()).detach(), 1).cpu()
            for i in range(len(result)):
                f.write(result[i]+'\n')
            v_accuracy += torch.sum(result == b_tag)

        v_accuracy = v_accuracy.float() / len(validloader.dataset) * 100
        print('Accuracy for Validation Set: ', v_accuracy)
            
    print('Total training:{:.1f}'.format(time.time() - start_time))
