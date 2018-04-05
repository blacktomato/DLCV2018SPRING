#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : bag_of_visual_word.py
 # Purpose : Recognition with Bag of Visual Word
 # Creation Date : 廿十八年四月四日 (週三) 廿二時廿九分十六秒
 # Last Modified : 廿十八年四月四日 (週三) 廿二時35分九秒
 # Created By : SL Chung
##############################################################

import cv2
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

