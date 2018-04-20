# -*- coding: utf-8 -*-
import numpy as np
from numpy import zeros,arange
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.pyplot import twinx
from math import ceil
import cv2

relative_path = "/home/lilin/my_code"
caffe_root = relative_path + '/deeptemporal/lib/caffe-tpp-net-python/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+' ')
    file.write('\n')
    file.close()

# savefile = 'frame_feat_flow.txt'
# if os.path.isfile(savefile):
#     os.remove(savefile)
#
# solver = caffe.SGDSolver(relative_path + '/deeptemporal/models/ucf101/flow_feat_solver.prototxt')
# solver.net.copy_from(relative_path + "/ucf101_split_1_rgb_flow_models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel")

savefile = 'frame_feat_rgb.txt'
if os.path.isfile(savefile):
    os.remove(savefile)

solver = caffe.SGDSolver(relative_path + '/deeptemporal/models/ucf101/rgb_feat_solver.prototxt')
solver.net.copy_from(relative_path + "/ucf101_split_1_rgb_flow_models/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel")


test_iter = 3783
savefeat = [[] for row in range(test_iter)]
solver.step(1)

for it in range(test_iter):
    print "Iteration: ", it
    solver.test_nets[0].forward()
    for i in range(25):
        feat = solver.test_nets[0].blobs['global_pool'].data[i].copy()
        feat = feat.reshape(1024)
        label = solver.test_nets[0].blobs['label'].data
        label = label.reshape(1)
        # print label
        feat_zip = np.concatenate((np.array(feat), np.array(label)))
        text_save(feat_zip, savefile)