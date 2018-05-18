#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : read_tb_event.py
 # Purpose : Export the image from the tensorboard event
 # Creation Date : 2018年05月14日 (週一) 16時16分13秒
 # Last Modified : 2018年05月18日 (週五) 19時34分23秒
 # Created By : SL Chung
##############################################################
import os
import sys
import scipy.misc
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def save_images_from_event(fn, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1  

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

event_acc = EventAccumulator(sys.argv[1]+'/VAE/')
event_acc.Reload()
_, step_nums, vals = zip(*event_acc.Scalars('MSE'))
step_M = np.asarray(step_nums)
MSE  = np.asarray(vals)
_, step_nums, vals = zip(*event_acc.Scalars('KLD'))
step_K = np.asarray(step_nums)
KLD  = np.asarray(vals)

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("MSE")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.plot(step_M, MSE, '-' ,label='MSE', c=[1,0,0] )

plt.subplot(1,2,2)
plt.title("KLD")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.plot(step_K, KLD, '-' ,label='KLD', c=[0,1,0] )
print('Output: MSE & KLD figs'  )
sys.stdout.flush()
plt.savefig("fig1_2.png")

event_acc = EventAccumulator(sys.argv[1]+'/GAN_DR/')
event_acc.Reload()
_, step_nums, vals = zip(*event_acc.Scalars('Loss_of_Discriminator'))
step_DR = np.asarray(step_nums)
DR = np.asarray(vals)

event_acc = EventAccumulator(sys.argv[1]+'/GAN_DF/')
event_acc.Reload()
_, step_nums, vals = zip(*event_acc.Scalars('Loss_of_Discriminator'))
step_DF = np.asarray(step_nums)
DF  = np.asarray(vals)

event_acc = EventAccumulator(sys.argv[1]+'/GAN_G/')
event_acc.Reload()
_, step_nums, vals = zip(*event_acc.Scalars('Loss_of_Generator'))
step_G = np.asarray(step_nums)
G  = np.asarray(vals)

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Discriminator")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid()
plt.plot(step_DR, DR, '-' ,c=[1,0.5,0.5] )
plt.plot(step_DF, DF, '-' ,c=[0.5,1,0.5] )
plt.plot(step_DR, smooth(DR,10), '-' ,label='Real', c=[1,0,0] )
plt.plot(step_DF, smooth(DF,10), '-' ,label='Fake', c=[0,1,0] )
plt.legend()

plt.subplot(1,2,2)
plt.title("Generator")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid()
plt.plot(step_G, G, '-' , c=[0,1,0] )
print('Output: MSE & KLD figs'  )
sys.stdout.flush()
plt.savefig("fig2_2.png")

event_acc = EventAccumulator(sys.argv[1]+'/GAN_DR/')
event_acc.Reload()
_, step_nums, vals = zip(*event_acc.Scalars('Loss_of_Discriminator'))
step_DR = np.asarray(step_nums)
DR = np.asarray(vals)

event_acc = EventAccumulator(sys.argv[1]+'/GAN_DF/')
event_acc.Reload()
_, step_nums, vals = zip(*event_acc.Scalars('Loss_of_Discriminator'))
step_DF = np.asarray(step_nums)
DF  = np.asarray(vals)

event_acc = EventAccumulator(sys.argv[1]+'/GAN_G/')
event_acc.Reload()
_, step_nums, vals = zip(*event_acc.Scalars('Loss_of_Generator'))
step_G = np.asarray(step_nums)
G  = np.asarray(vals)
