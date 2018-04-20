import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix

relative_path = "/data4/lilin/zjg_code"

caffe_root = relative_path + "/temporal-segment-networks/lib/caffe-action/"
sys.path.insert(0, caffe_root + 'python')


from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file

from pyActionRecog.utils.video_funcs import default_aggregation_func, ave_aggregation_func, sci_crop_aggregation_func, sci_stream_aggregation_func, sci_crop_stream_aggregation_func

num_scores = 1
# nn = float(7)/13
# print nn
#
# r = np.load("/data/lilin/lilin_code/zjg_code/tle/ucf101_split_1_rgb_seg124_3dropout_imagenet_stage_2_lr_0.0001_iter_600.npz")
# r = np.load("/data/lilin/lilin_code/zjg_code/tle/ucf101_split_1_rgb_seg124_3dropout_stage_2_lr_0.00001_iter_2100.npz")
# video_scores_1 = r['scores']
# #video_scores_1 = video_scores_1.reshape(3783,10,101)
# video_labels_1 = r['labels']

r = np.load("/data/lilin/lilin_code/zjg_code/tle/hmdb51_split_1_tsn_flow_score.npz")
video_scores = r['scores']
video_labels = r['labels']

def gated_fusion(i):
    video_pred = [np.argmax(default_aggregation_func(x[i], normalization=False)) for x in video_scores]
    video_labels = [x[num_scores] for x in video_scores]

    cf = confusion_matrix(video_labels, video_pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit/cls_cnt
    print cls_acc
    print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)

def ave_fusion(i, j, weight = 1):
    video_pred = [np.argmax(ave_aggregation_func(x[i], x[j], weight = weight)) for x in video_scores]
    video_labels = [x[num_scores] for x in video_scores]

    cf = confusion_matrix(video_labels, video_pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit/cls_cnt
    print cls_acc
    print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)

def ave_fusion(x, y, weight = 1):
    video_pred = [np.argmax(ave_aggregation_func(x[i], x[j], weight = weight)) for x in video_scores]
    video_labels = [x[num_scores] for x in video_scores]

    cf = confusion_matrix(video_labels, video_pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit/cls_cnt
    print cls_acc
    print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)



def sci_fusion_crop(i, j):
    video_pred = [np.argmax(sci_crop_aggregation_func(x[i], x[j])) for x in video_scores]
    video_labels = [x[num_scores] for x in video_scores]

    cf = confusion_matrix(video_labels, video_pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit/cls_cnt
    print cls_acc
    print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)

def sci_fusion_stream(i, j):
    video_pred = [np.argmax(sci_stream_aggregation_func(x[i], x[j])) for x in video_scores]
    video_labels = [x[num_scores] for x in video_scores]
    cf = confusion_matrix(video_labels, video_pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit/cls_cnt
    print cls_acc
    print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)

def sci_fusion_crop_stream(i, j):
    video_pred = [np.argmax(sci_crop_stream_aggregation_func(x[i], x[j])) for x in video_scores]
    video_labels = [x[num_scores] for x in video_scores]

    cf = confusion_matrix(video_labels, video_pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit/cls_cnt
    print cls_acc
    print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)



print "Gated fusion\n"
gated_fusion(0)
# print "ave fusion\n"
# ave_fusion(1, 2)
# print "weighted ave fusion\n"
# ave_fusion(1, 2, weight = 2)
#
# print "sci fusion crop\n"
# sci_fusion_crop(1, 2)
# #
# print "sci fusion stream \n"
# sci_fusion_stream(1, 2)
#
# print "sci fusion crop stream \n"
# sci_fusion_crop_stream(1, 2)