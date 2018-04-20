import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix
import argparse
import os
import sys
import math
import cv2
import glob
import fnmatch
import os
import random
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix


relative_path = "/home/lilin/my_code"
caffe_root = relative_path + '/deeptemporal/lib/caffe-tpp-net-python/'
sys.path.insert(0, caffe_root + 'python')

from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file

from pyActionRecog.utils.video_funcs import default_aggregation_func

net_weights = "/home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112.caffemodel"

split = int(net_weights.split('split')[1].split('_')[1])

str = net_weights.split('/')[-1]

if 'rgb' in str:
   modality = 'rgb'
else:
   modality = 'flow'

if 'ucf101' in str:
    dataset = 'ucf101'
else:
    dataset = 'hmdb51'

if dataset == 'ucf101':
    frame_path = relative_path + "/UCF-101-result/"
else:
    frame_path = relative_path + "/HMDB-51-result/"


from pyActionRecog.action_caffe import CaffeNet

# build neccessary information
print dataset
split_tp = parse_split_file(dataset)
# f_info = parse_directory(frame_path,
#                          rgb_prefix, flow_x_prefix, flow_y_prefix)


for i in xrange(1,4):
    eval_video_list = split_tp[i - 1][1]
    file = open('data/hmdb51_splits/testlist0{}.txt'.format(i), 'w')
    def eval_video(video):
        global net
        label = video[1]
        vid = video[0]
        print vid
        file.write(vid + ".avi")
        file.write('\n')
    video_scores = map(eval_video, eval_video_list)
