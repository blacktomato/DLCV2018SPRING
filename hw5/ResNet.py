#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : ResNet.py
 # Purpose : Classification with ResNet feature
 # Creation Date : 廿十八年五月廿四日 (週四) 十五時廿一分九秒
 # Last Modified : 廿十八年五月廿四日 (週四) 十七時廿八分十四秒
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
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import torchvision.models as models

from reader import readShortVideo, getVideoList 

#resnet50 = models.resnet50(pretrained = True)

class FC_for_C(nn.Module):
    def __init__(self, feature_extractor):
        super(FC_for_C, self).__init__()
        self.feature_extractor = feature_extractor
         
    def FC(self, features):
        
    def forward(self, images):
        x = feature_extractor(images)
        return FC(x)

