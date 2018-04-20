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

savefile = 'tpp_feat_flow.txt'
if os.path.isfile(savefile):
    os.remove(savefile)

solver = caffe.SGDSolver(relative_path + '/deeptemporal/models/ucf101/flow_feat_tpp_solver.prototxt')
solver.net.copy_from(relative_path + "/ucf101_split_1_rgb_flow_models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel")

# savefile = 'end2end_tpp_feat_flow.txt'
# if os.path.isfile(savefile):
#     os.remove(savefile)
#
# solver = caffe.SGDSolver(relative_path + '/deeptemporal/models/ucf101/flow_feat_tpp_solver.prototxt')
# solver.net.copy_from("/home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_flow_tpp_delete_dropout_lr_0.00001_iter_1500.caffemodel")

# savefile = 'tpp_feat_rgb.txt'
# if os.path.isfile(savefile):
#     os.remove(savefile)
# solver = caffe.SGDSolver(relative_path + '/deeptemporal/models/ucf101/rgb_feat_tpp_solver.prototxt')
# solver.net.copy_from(relative_path + "/ucf101_split_1_rgb_flow_models/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel")


# savefile = 'end2end_tpp_feat_rgb.txt'
# if os.path.isfile(savefile):
#     os.remove(savefile)
# solver = caffe.SGDSolver(relative_path + '/deeptemporal/models/ucf101/rgb_feat_tpp_solver.prototxt')
# solver.net.copy_from("/home/lilin/my_code/ucf101_split_1_rgb_flow_models/ucf101_split_1_rgb_tpp_delete_dropout_lr_0.00001_iter_600.caffemodel")


test_iter = 3783
savefeat = [[] for row in range(test_iter)]
solver.step(1)

for it in range(test_iter):
    print "Iteration: ", it
    solver.test_nets[0].forward()
    feat = solver.test_nets[0].blobs['pyramid_dropout'].data[0].copy()
    feat = feat.reshape(7168)
    label = solver.test_nets[0].blobs['label'].data
    label = label.reshape(1)
    print label
    feat_zip = np.concatenate((np.array(feat), np.array(label)))
    text_save(feat_zip, savefile)


    #savefeat[it].append(np.array(feat_zip))
    #vis_square(feat)