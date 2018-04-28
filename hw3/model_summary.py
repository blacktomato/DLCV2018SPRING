#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : FCN_test.py
 # Purpose : Loading the FCN Model and Test the Data
 # Creation Date : 廿十八年四月廿六日 (週四) 廿一時廿三分三秒
 # Last Modified : 2018年04月28日 (週六) 22時59分34秒
 # Created By : SL Chung
##############################################################
import sys
import os
import numpy as np
import time
from keras.models import Model, load_model
# GPU setting\n",
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

model_name = sys.argv[1]
fcn_model = load_model(model_name)
fcn_model.summary()

