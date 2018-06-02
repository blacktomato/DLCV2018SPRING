#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : RNN_test.py
 # Purpose : Test the RNN model
 # Creation Date : 2018年06月02日 (週六) 14時48分29秒
 # Last Modified : 2018年06月02日 (週六) 16時15分46秒
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
import torch.nn.utils as nn_utils
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import torchvision.models as models

from reader import readShortVideo, getVideoList 
from RNN import *

resnet50 = models.resnet50(pretrained = True)
resnet50.cuda() 
resnet50.eval()

if __name__=='__main__': 
    start_time = time.time()
    batch_size = 100

    video_info = getVideoList(sys.argv[3])
    video_path = sys.argv[2] 
    video_category = video_info['Video_category']
    video_name = video_info['Video_name']
    video_tag = np.array(video_info['Action_labels']).astype('float') 
    video_tag = torch.from_numpy(video_tag)
    del video_info

    video_ts, video_len = Video2Seq(video_path, video_category, video_name)
    video_set = Data.TensorDataset(video_ts, torch.Tensor(video_len).long(), video_tag.long())
    videoloader = Data.DataLoader(dataset=video_set, batch_size=batch_size) 
     
    rnn = torch.load(sys.argv[1]) 
    rnn.cuda()

    #Testing
    rnn.eval()
    v_accuracy = 0
    h_state = None
    
    with open(os.path.join(sys.argv[4], 'p2_result.txt'), 'w+') as f:
        for batch_idx, (b_feature, b_len, b_tag) in enumerate(videoloader):
            seq_len = b_len.tolist()
            sort_index = np.argsort(seq_len)[::-1]
            b_feature = b_feature[sort_index.tolist(), 0:max(seq_len), :]
            b_tag = b_tag[sort_index.tolist()]
            pack = nn_utils.rnn.pack_padded_sequence(b_feature.cuda(),
                            np.sort(seq_len)[::-1], batch_first=True)

            result = rnn(pack, h_state)[0].detach()
            result = torch.argmax(result, 1).cpu()
            v_accuracy += torch.sum(result == b_tag)
           
            #unsort the result
            result = result[np.argsort(sort_index)].numpy()
            for i in range(len(result)):
                f.write(str(result[i])+'\n')

        v_accuracy = v_accuracy.float() / len(videoloader.dataset) * 100

    print('Accuracy:', v_accuracy.item())
    print('Total time:{:.1f}'.format(time.time() - start_time))

