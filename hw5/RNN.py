#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : RNN.py
 # Purpose : Use RNN structure to classify the video 
 # Creation Date : 2018年05月30日 (週三) 15時44分46秒
 # Last Modified : 2018年06月02日 (週六) 15時02分46秒
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

resnet50 = models.resnet50(pretrained = True)
resnet50.cuda() 
resnet50.eval()

class RNN(nn.Module):
    def __init__(self, n_classes, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size = 1000,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size/2)
        self.bn1 = nn.BatchNorm1d(hidden_size/2)
        self.fc2 = nn.Linear(hidden_size/2, hidden_size/4)
        self.bn2 = nn.BatchNorm1d(hidden_size/4)
        self.fc3 = nn.Linear(hidden_size/4, hidden_size/8)
        self.bn3 = nn.BatchNorm1d(hidden_size/8)
        self.fc4 = nn.Linear(hidden_size/8, n_classes)
         
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, features, h_state):
        x, h_state = self.rnn(features, h_state)
        x, l = nn_utils.rnn.pad_packed_sequence(x, batch_first=True)
        pos = [(i * x.size(1) + l.tolist()[i] - 1) for i in range(x.size(0))]
        x = x.cpu().view(-1, self.hidden_size).cuda()[pos]
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        return F.softmax(self.fc4(x), dim=-1), h_state

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def Video2Seq(video_path, video_category, video_name):
    features = torch.Tensor()
    seq_length = []
    for i in range(len(video_name)):
        frames = readShortVideo(video_path, video_category[i], video_name[i])
        ts_frames = torch.from_numpy(frames.transpose((0, 3, 1, 2))).float()/ 255.
        sys.stdout.write('\rReading the Video... : {:}'.format(i))
        sys.stdout.flush()

        set = Data.TensorDataset(ts_frames)
        dataloader = Data.DataLoader(dataset=set, 
                                    batch_size=3) 

        seq_length.append(0)
        for batch_idx, b_frame in enumerate(dataloader):
            features = torch.cat([features, resnet50(b_frame[0].cuda()).detach().cpu()])
            seq_length[i] += len(b_frame[0]) 

    max_length = max(seq_length)
    seq = torch.zeros(len(seq_length), max_length, features.shape[1])
    start = 0
    
    for i in range(len(seq_length)):
        seq[i,0:seq_length[i],:] = features[start:start+seq_length[i],:]
        start += seq_length[i]

    sys.stdout.write('... Done\n')
    sys.stdout.flush()
    return seq, seq_length 

if __name__=='__main__': 
    epochs = 100
    n_classes = 11
    hidden_size = 1000
    batch_size = 100
    num_layers = 1
    boardX = True
    presave_tensor = True

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
    

    valid_info = getVideoList('/data/r06942052/HW5_data/TrimmedVideos/label/gt_valid.csv')
    valid_path = '/data/r06942052/HW5_data/TrimmedVideos/video/valid'
    valid_category = valid_info['Video_category']
    valid_name = valid_info['Video_name']
    valid_tag = np.array(valid_info['Action_labels']).astype('float') 
    valid_tag = torch.from_numpy(valid_tag)
    del valid_info

    if presave_tensor:
        train_ts = torch.load('/data/r06942052/rnn_train_ts.pt')
        valid_ts = torch.load('/data/r06942052/rnn_valid_ts.pt')
        train_len = torch.load('/data/r06942052/rnn_train_len.pt')
        valid_len = torch.load('/data/r06942052/rnn_valid_len.pt')
    else: 
        train_ts, train_len = Video2Seq(train_path, train_category, train_name)
        valid_ts, valid_len = Video2Seq(valid_path, valid_category, valid_name)
        torch.save(train_ts, '/data/r06942052/rnn_train_ts.pt')
        torch.save(valid_ts, '/data/r06942052/rnn_valid_ts.pt')
        torch.save(train_len, '/data/r06942052/rnn_train_len.pt')
        torch.save(valid_len, '/data/r06942052/rnn_valid_len.pt')

    train_set = Data.TensorDataset(train_ts, torch.Tensor(train_len).long(), train_tag.long())
    valid_set = Data.TensorDataset(valid_ts, torch.Tensor(valid_len).long(), valid_tag.long())

    trainloader = Data.DataLoader(dataset=train_set, batch_size=batch_size) 
    validloader = Data.DataLoader(dataset=valid_set, batch_size=batch_size) 
     
    rnn = RNN(n_classes, hidden_size, num_layers)
    rnn.cuda()
    rnn.weight_init(mean=0.0, std=0.02)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(rnn.parameters(), lr=1e-4, betas=(0.5,0.999))

    train_status = '\rEpoch:  {} [{}/{}] Train Loss: {:.3f} '
    valid_status = 'Valid Percentage: {:.3f}% ---> '

    training_time = time.time()
    for epoch in range(epochs): 
        start_time = time.time()
        h_state = None
        rnn.train()
        for batch_idx, (b_feature, b_len, b_tag) in enumerate(trainloader):
            seq_len = b_len.tolist()
            sort_index = np.argsort(seq_len)[::-1]
            b_feature = b_feature[sort_index.tolist(), 0:max(seq_len), :]
            b_tag = b_tag[sort_index.tolist()]
            pack = nn_utils.rnn.pack_padded_sequence(b_feature.cuda(),
                            np.sort(seq_len)[::-1], batch_first=True)
            
            optimizer.zero_grad()
            result, _ = rnn(pack, h_state)
            train_loss = criterion(result, b_tag.cuda()) 
            train_loss.backward()
            optimizer.step()

            if boardX:
                writer.add_scalar('Train Loss', 
                train_loss,
                epoch*len(trainloader.dataset)/batch_size+batch_idx)
            sys.stdout.write(train_status.format( 
                epoch+1, batch_idx * batch_size + len(b_tag), len(trainloader.dataset), 
                train_loss))
        #Validation
        rnn.eval()
        v_accuracy = 0
        h_state = None
        for batch_idx, (b_feature, b_len, b_tag) in enumerate(validloader):
            seq_len = b_len.tolist()
            sort_index = np.argsort(seq_len)[::-1]
            b_feature = b_feature[sort_index.tolist(), 0:max(seq_len), :]
            b_tag = b_tag[sort_index.tolist()]
            pack = nn_utils.rnn.pack_padded_sequence(b_feature.cuda(),
                            np.sort(seq_len)[::-1], batch_first=True)

            result = rnn(pack, h_state)[0].detach()
            valid_loss = criterion(result, b_tag.cuda())
            v_accuracy += torch.sum(torch.argmax(result, 1).cpu() == b_tag)
            if boardX:
                writer.add_scalar('Valid Loss', 
                valid_loss,
                epoch*len(trainloader.dataset)/batch_size+batch_idx)

        v_accuracy = v_accuracy.float() / len(validloader.dataset) * 100
        if boardX:
            writer.add_scalar('Valid Accuracy', 
            v_accuracy,
            epoch*len(trainloader.dataset)/batch_size)
        sys.stdout.write(valid_status.format(v_accuracy))
        sys.stdout.write('Times:{:.1f}\n'.format(time.time() - start_time))

    print('Total training:{:.1f}'.format(time.time() - training_time))
    print('Saving model...')
    torch.save(rnn, 'RNN.pt')
