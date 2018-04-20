# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:22:22 2016

@author: root
"""
import numpy as np
from numpy import zeros,arange
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.pyplot import twinx
from math import ceil
import cv2

caffe_root = '/home/zhujiagang/temporal-segment-networks/lib/caffe-center-loss/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(1)
caffe.set_mode_gpu()


def vis_square(data):
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data);
    plt.axis('off')

def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+' ')
    file.write('\n')
    file.close()


if os.path.isfile('feat_flow.txt'):
    os.remove('feat_flow.txt')
# if os.path.isfile('feat_rgb.txt'):
#     os.remove('feat_rgb.txt')
# if os.path.isfile('feat_gating.txt'):
#     os.remove('feat_gating.txt')


solver = caffe.SGDSolver('/home/zhujiagang/temporal-segment-networks/models/ucf101/flow_feat_solver.prototxt')
solver.net.copy_from("/home/zhujiagang/temporal-segment-networks/models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel")
test_iter = 69
savefeat = [[] for row in range(test_iter)]
solver.step(1)

# 进行解算
for it in range(test_iter):
    solver.test_nets[0].forward()
    # feat_flow = solver.test_nets[0].blobs['flow_global_pool'].data[0].copy()
    # feat_flow = feat_flow.reshape(1024)
    # print feat_flow.shape

    feat_rgb = solver.test_nets[0].blobs['global_pool'].data[0].copy()
    feat_rgb = feat_rgb.reshape(1024)
    #print feat_rgb.shape
    feat_rgb_1 = solver.test_nets[0].blobs['global_pool'].data[1].copy()
    feat_rgb_1 = feat_rgb_1.reshape(1024)
    #print feat_rgb_1.shape
    feat_rgb_2 = solver.test_nets[0].blobs['global_pool'].data[2].copy()
    feat_rgb_2 = feat_rgb_2.reshape(1024)
    #print feat_rgb_2.shape

    output = solver.test_nets[0].blobs['pool_fc'].data.argmax()
    output = output.reshape(1)

    # feat_gating = solver.test_nets[0].blobs['gating_global_pool'].data[0].copy()
    # feat_gating = feat_gating.reshape(1024)
    # print feat_gating.shape
    #
    label = solver.test_nets[0].blobs['label'].data
    label = label.reshape(1)
    #print label.shape

    print label, output
    # feat_flow_zip = np.concatenate((np.array(feat_flow), np.array(label)))
    # text_save(feat_flow_zip, 'feat_flow.txt')

    feat_rgb_zip = np.concatenate((np.array(feat_rgb), np.array(label), np.array(output)))
    text_save(feat_rgb_zip, 'feat_flow.txt')
    feat_rgb_zip = np.concatenate((np.array(feat_rgb_1), np.array(label), np.array(output)))
    text_save(feat_rgb_zip, 'feat_flow.txt')
    feat_rgb_zip = np.concatenate((np.array(feat_rgb_2), np.array(label), np.array(output)))
    text_save(feat_rgb_zip, 'feat_flow.txt')
    # feat_gating_zip = np.concatenate((np.array(feat_gating), np.array(label)))
    # text_save(feat_gating_zip, 'feat_gating.txt')

    #savefeat[it].append(np.array(feat_zip))
    #vis_square(feat)