#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : color_testure_segment.py
 # Purpose : Color & Texture Segmentation for images
 # Creation Date : 廿十八年三月廿八日 (週三) 十五時41分十秒
 # Last Modified : 廿十八年三月廿八日 (週三) 十五時46分二秒
 # Created By : SL Chung
##############################################################

import cv2
import numpy as np
import scipy.io as scio

filter = scio.loadmat("../data/hw2_dataset/Problem2/filterBank.mat")
F = filter['F']
