#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : RNN_S2S.py
 # Purpose : Train a RNN Model for Sequence to Sequence Task
 # Creation Date : 2018年06月02日 (週六) 16時09分03秒
 # Last Modified : 2018年06月08日 (週五) 01時21分17秒
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

from reader import readImgSeq

resnet50 = models.resnet50(pretrained = True)
resnet50.cuda() 
resnet50.eval()

class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers, bidirectional):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size = 1000,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = bidirectional
        )
         
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, features, h_state):
        x, h_state = self.rnn(features, h_state)
        return x

class CNN(nn.Module):
    def __init__(self, n_classes, input_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, input_size/2)
        self.bn1 = nn.BatchNorm1d(input_size/2)
        self.fc2 = nn.Linear(input_size/2, input_size/4)
        self.bn2 = nn.BatchNorm1d(input_size/4)
        self.fc3 = nn.Linear(input_size/4, input_size/8)
        self.bn3 = nn.BatchNorm1d(input_size/8)
        self.fc4 = nn.Linear(input_size/8, n_classes)
         
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, feature):
        x = F.leaky_relu(self.bn1(self.fc1(feature)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        return F.softmax(self.fc4(x), dim=-1)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def Videos2Seqs(video_path):
    seqs = []
    dirs = [dir for dir in os.listdir(video_path)]
    dirs.sort()
    for i, dir in enumerate(dirs): 
        frames = readImgSeq(os.path.join(video_path, dir))
        ts_frames = torch.from_numpy(frames.transpose((0, 3, 1, 2))).float()/ 255.
        sys.stdout.write('\rReading the Video... : {:}'.format(i+1))
        sys.stdout.flush()

        set = Data.TensorDataset(ts_frames)
        dataloader = Data.DataLoader(dataset=set, 
                                    batch_size=3) 
        features = torch.Tensor()
        for batch_idx, b_frame in enumerate(dataloader):
            features = torch.cat([features, resnet50(b_frame[0].cuda()).detach().cpu()])
        seqs.append(features) 

    sys.stdout.write('... Done\n')
    sys.stdout.flush()
    return seqs

def readLabel(label_path):
    labels = []
    txts = [txt for txt in os.listdir(label_path)]
    txts.sort()
    for i, txt in enumerate(txts): 
        with open(os.path.join(label_path, txt), 'r') as f:
            label = f.read().splitlines()
        label = torch.from_numpy(np.array(label).astype('float'))
        sys.stdout.write('\rReading the Label... : {:}'.format(i+1))
        sys.stdout.flush()
        labels.append(label)

    sys.stdout.write('... Done\n')
    sys.stdout.flush()
    return labels

if __name__=='__main__': 
    epochs = 100
    n_classes = 11
    bidirectional = True 
    hidden_size = 1000
    input_size = hidden_size
    batch_size = 20
    num_layers = 1

    time_steps = 400
    segment_steps = 20
    boardX = True
    presave_tensor = True 
    presave_dataloader = True
    if bidirectional:
        input_size *= 2

    if boardX:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('runs/'+sys.argv[1])

    train_path = '/data/r06942052/HW5_data/FullLengthVideos/videos/train'
    train_tag_l = readLabel('/data/r06942052/HW5_data/FullLengthVideos/labels/train') 

    valid_path = '/data/r06942052/HW5_data/FullLengthVideos/videos/valid'
    valid_tag_l = readLabel('/data/r06942052/HW5_data/FullLengthVideos/labels/valid') 

    if presave_tensor:
        train_ts_l = torch.load('/data/r06942052/s2s_train_ts.pt')
        valid_ts_l = torch.load('/data/r06942052/s2s_valid_ts.pt')
    else: 
        train_ts_l = Videos2Seqs(train_path)
        valid_ts_l = Videos2Seqs(valid_path)
        torch.save(train_ts_l, '/data/r06942052/s2s_train_ts.pt')
        torch.save(valid_ts_l, '/data/r06942052/s2s_valid_ts.pt')
    
    if presave_dataloader:
        trainloader = torch.load('/data/r06942052/s2s_ts400_s20_b20_trainloader.pt')
    else:
        train_ts = []
        train_tag = []
        for i, video in enumerate(train_ts_l): #list
            train_ts.append(torch.Tensor())
            train_tag.append(torch.Tensor())
            if len(video) >= time_steps:
                starts = np.arange(0, len(video) - time_steps + 1, segment_steps)
                for start in starts:
                    train_ts[i] = torch.cat([train_ts[i], video[start:start+time_steps]])
                    train_tag[i] = torch.cat([train_tag[i], train_tag_l[i][start:start+time_steps].float()])
                    sys.stdout.write('\rSampling the Sequence... : {:}_{:.2f}%'.format(
                                                                i+1, start*100.0/len(video)))
                    sys.stdout.flush()

        train_ts = torch.cat(train_ts)
        train_tag = torch.cat(train_tag)
        train_ts = train_ts.view(-1, time_steps, 1000)
        train_tag = train_tag.view(-1, time_steps).long()
        train_len = [time_steps]*train_ts.size(0)

        train_set = Data.TensorDataset(train_ts, torch.Tensor(train_len).long(), train_tag)

        trainloader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True) 
        torch.save(trainloader, '/data/r06942052/s2s_ts400_s20_b20_trainloader.pt')
     

    rnn = RNN(hidden_size, num_layers, bidirectional)
    cnn = CNN(n_classes, input_size)
    rnn.cuda()
    cnn.cuda()
    rnn.weight_init(mean=0.0, std=0.02)
    cnn.weight_init(mean=0.0, std=0.02)

    criterion = nn.CrossEntropyLoss().cuda()
    R_optimizer = optim.Adam(rnn.parameters(), lr=1e-4, betas=(0.5,0.999))
    C_optimizer = optim.Adam(cnn.parameters(), lr=1e-4, betas=(0.5,0.999))

    train_status = '\rEpoch:  {} [{}/{}] Train Loss: {:.3f} '
    valid_status = 'Valid: {:.3f}% ---> '

    training_time = time.time()
    for epoch in range(epochs): 
        start_time = time.time()
        h_state = None
        rnn.train()
        cnn.train()
        for batch_idx, (b_feature, b_len, b_tag) in enumerate(trainloader):
            pack = nn_utils.rnn.pack_padded_sequence(b_feature.cuda(), b_len, batch_first=True)
            
            R_optimizer.zero_grad()
            C_optimizer.zero_grad()
            h = rnn(pack, h_state)
            h, l = nn_utils.rnn.pad_packed_sequence(h, batch_first=True)
            h = h.cpu().view(-1, input_size).cuda()
            result = cnn(h)
            train_loss = criterion(result, b_tag.view(-1).cuda()) 
            train_loss.backward()
            R_optimizer.step()
            C_optimizer.step()

            if boardX:
                writer.add_scalar('Train Loss', 
                train_loss,
                epoch*len(trainloader.dataset)/batch_size+batch_idx)
            sys.stdout.write(train_status.format( 
                epoch+1, batch_idx * batch_size + len(b_tag), len(trainloader.dataset), 
                train_loss))
                
        #Validation
        rnn.eval()
        cnn.eval()
        v_accuracy = 0
        valid_loss = 0
        h_state = None
        total = 0
        for i in range(len(valid_ts_l)):
            total += len(valid_ts_l[i])

            h = rnn(valid_ts_l[i].view(1, -1, 1000).cuda(), h_state).detach()
            h = h.cpu().view(-1, input_size).cuda()
            result = cnn(h).detach()

            valid_loss += criterion(result, valid_tag_l[i].long().cuda())
            v_accuracy += torch.sum(torch.argmax(result, 1).cpu() == valid_tag_l[i].long())

        if boardX:
            writer.add_scalar('Valid Loss', 
            valid_loss / len(valid_ts_l),
            epoch*len(trainloader.dataset)/batch_size)

        v_accuracy = v_accuracy.float() / total * 100
        if boardX:
            writer.add_scalar('Valid Accuracy', 
            v_accuracy,
            epoch*len(trainloader.dataset)/batch_size)
        sys.stdout.write(valid_status.format(v_accuracy))
        sys.stdout.write('Times:{:.1f}\n'.format(time.time() - start_time))

    print('Total training: {:.1f} second'.format(time.time() - training_time))
    print('Saving model...')
    torch.save(rnn, 'RNN_S2S.pt')
    torch.save(cnn, 'CNN_S2S.pt')
