#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : ResNet.py
 # Purpose : Classification with ResNet feature
 # Creation Date : 廿十八年五月廿四日 (週四) 十五時廿一分九秒
 # Last Modified : 2018年05月31日 (週四) 19時20分06秒
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
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import torchvision.models as models

from reader import readShortVideo, getVideoList 

resnet50 = models.resnet50(pretrained = True)
resnet50.cuda() 
resnet50.eval()

class FC_for_C(nn.Module):
    def __init__(self, n_classes):
        super(FC_for_C, self).__init__()
        self.fc1 = nn.Linear(1000, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.drop1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(250, 125)
        self.bn3 = nn.BatchNorm1d(125)
        self.drop2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(125, n_classes)
         
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, features):
        x = F.leaky_relu(self.bn1(self.fc1(features)))
        x = F.leaky_relu(self.drop1(self.bn2(self.fc2(x))))
        x = F.leaky_relu(self.drop2(self.bn3(self.fc3(x))))
        return F.softmax(self.fc4(x), dim=-1)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def Video2Tensor(video_path, video_category, video_name):
    features = torch.Tensor()
    for i in range(len(video_name)):
        frames = readShortVideo(video_path, video_category[i], video_name[i])
        ts_frames = torch.from_numpy(frames.transpose((0, 3, 1, 2))).float()/ 255.
        sys.stdout.write('\rReading the Video... Frame: {:}'.format(i))
        sys.stdout.flush()
        set = Data.TensorDataset(ts_frames)

        dataloader = Data.DataLoader(dataset=set, 
                                    batch_size=1) 
        feature = torch.zeros(1,1000).cuda()
        for batch_idx, b_frame in enumerate(dataloader):
            feature += resnet50(b_frame[0].cuda()).detach()
        features = torch.cat([features, (feature/len(set)).cpu()])
    sys.stdout.write('... Done\n')
    sys.stdout.flush()
    return features

if __name__=='__main__': 
    epochs = 2000
    n_classes = 11
    batch_size = 100
    boardX = True

    if boardX:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('runs/'+sys.argv[1])
    train_info = getVideoList('/data/r06942052/HW5_data/TrimmedVideos/label/gt_train.csv')
    train_path = '/data/r06942052/HW5_data/TrimmedVideos/video/train'
    train_category = train_info['Video_category']
    train_name = train_info['Video_name']
    train_tag = np.array(train_info['Action_labels']).astype('float') 
    train_tag = torch.from_numpy(train_tag)
    del train_info
    '''
    train_ts = Video2Tensor(train_path, train_category, train_name)
    torch.save(train_ts, '/data/r06942052/train_ts.pt')
    '''
    valid_info = getVideoList('/data/r06942052/HW5_data/TrimmedVideos/label/gt_valid.csv')
    valid_path = '/data/r06942052/HW5_data/TrimmedVideos/video/valid'
    valid_category = valid_info['Video_category']
    valid_name = valid_info['Video_name']
    valid_tag = np.array(valid_info['Action_labels']).astype('float') 
    valid_tag = torch.from_numpy(valid_tag)
    del valid_info
    '''
    valid_ts = Video2Tensor(valid_path, valid_category, valid_name)
    torch.save(valid_ts, '/data/r06942052/valid_ts.pt')
    '''
    train_ts = torch.load('/data/r06942052/train_ts.pt')
    valid_ts = torch.load('/data/r06942052/valid_ts.pt')

    train_set = Data.TensorDataset(train_ts, train_tag.long())
    valid_set = Data.TensorDataset(valid_ts, valid_tag.long())

    trainloader = Data.DataLoader(dataset=train_set, batch_size=batch_size) 
    validloader = Data.DataLoader(dataset=valid_set, batch_size=batch_size) 

    Net = FC_for_C(n_classes)
    Net.cuda()
    Net.train()
    Net.weight_init(mean=0.0, std=0.02)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(Net.parameters(), lr=1e-4, betas=(0.5,0.999))

    train_status = '\rEpoch:  {} [{}/{}] Train Loss: {:.3f} '
    valid_status = 'Valid Percentage: {:.3f}% ---> '

    training_time = time.time()
    for epoch in range(epochs): 
        start_time = time.time()
        for batch_idx, (b_feature, b_tag) in enumerate(trainloader):
            optimizer.zero_grad()
            train_loss = criterion(Net(b_feature.cuda()), b_tag.cuda()) 
            train_loss.backward()
            optimizer.step()
            

            if boardX:
                writer.add_scalar('Train Loss', 
                train_loss,
                epoch*len(trainloader.dataset)/batch_size+batch_idx)
            sys.stdout.write(train_status.format( 
                epoch+1, batch_idx * batch_size + len(b_tag), len(trainloader.dataset), 
                train_loss))

        
        Net.eval()
        v_accuracy = 0
        for batch_idx, (b_feature, b_tag) in enumerate(validloader):
            result = torch.argmax(Net(b_feature.cuda()).detach(), 1).cpu()
            v_accuracy += torch.sum(result == b_tag)

        v_accuracy = v_accuracy.float() / len(validloader.dataset) * 100
        if boardX:
            writer.add_scalar('Valid Accuracy', 
            v_accuracy,
            epoch*len(trainloader.dataset)/batch_size)
        sys.stdout.write(valid_status.format(v_accuracy))
                
        sys.stdout.write('Times:{:.1f}\n'.format(time.time() - start_time))
    print('Total training:{:.1f}'.format(time.time() - training_time))
    print('Saving model...')
    torch.save(Net, 'FC_for_C.pt')
