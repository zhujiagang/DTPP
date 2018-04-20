import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
caffe_root = '/home/zhujiagang/temporal-segment-networks/lib/caffe-action/'
sys.path.insert(0, caffe_root + 'python')
from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file
from pyActionRecog.utils.video_funcs import default_aggregation_func

dataset ='ucf101'
split =1
modality ='flow'
frame_path = '/home/zjg/zjg/tsncaffe/UCF-101-result/'

net_proto = "/home/zhujiagang/temporal-segment-networks/models/ucf101/tsn_bn_inception_flow_deploy.prototxt"
net_weights = "/home/zhujiagang/temporal-segment-networks/models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel"
rgb_prefix ='img_'
flow_x_prefix ='flow_x_'
flow_y_prefix ='flow_y_'
num_frame_per_video = 1
save_scores = None
num_worker = 1
num_id = 1
gpus = [1]
from pyActionRecog.action_caffe import CaffeNet

# build neccessary information
print dataset
split_tp = parse_split_file(dataset)
f_info = parse_directory(frame_path, rgb_prefix, flow_x_prefix, flow_y_prefix)

gpu_list = None
eval_video_list = split_tp[split - 1][1]
score_name = 'prob'

def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if num_worker > 1 else 1

    if gpu_list is None:
        net = CaffeNet(net_proto, net_weights, my_id-1)
    else:
        net = CaffeNet(net_proto, net_weights, gpu_list[my_id - 1])

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

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

def eval_video(video):
    global net
    label = video[1]
    vid = video[0]
    video_frame_path = f_info[0][vid]

    cnt_indexer = 2
    frame_cnt = f_info[cnt_indexer][vid]
    stack_depth = 5

    if num_frame_per_video >= 2:
        step = (frame_cnt - stack_depth) / (num_frame_per_video-1)
        if step > 0:
            frame_ticks = range(1, min((2 + step * (num_frame_per_video-1)), frame_cnt+1), step)
        else:
            frame_ticks = [1] * num_frame_per_video
    else:
        if frame_cnt > stack_depth:
            frame_ticks = [(frame_cnt - stack_depth) / 2]
        else:
            frame_ticks = [frame_cnt / 2]

    assert(len(frame_ticks) == num_frame_per_video)
    frame_scores = []

    for tick in frame_ticks:
        ii = 0
        if modality == 'rgb':
            name = '{}{:05d}.jpg'.format(rgb_prefix, tick)
            frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
            scores = net.predict_single_frame([frame, ], score_name, frame_size=(340, 256))
            frame_scores.append(scores)
        if modality == 'flow':
            frame_idx = [min(frame_cnt, tick + offset) for offset in xrange(stack_depth)]
            flow_stack = []
            for idx in frame_idx:
                ii = ii + 1
                x_name = '{}{:05d}.jpg'.format(flow_x_prefix, idx)
                y_name = '{}{:05d}.jpg'.format(flow_y_prefix, idx)
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))

            scores, feat = net.predict_single_flow_stack_feature_map(flow_stack, score_name, frame_size=(224, 224))
            vis_square(feat)
            frame_scores.append(scores)

    print 'video {} done'.format(vid)
    sys.stdin.flush()
    return np.array(frame_scores), label

if num_worker > 1:
    pool = multiprocessing.Pool(num_worker, initializer=build_net)
    video_scores = pool.map(eval_video, eval_video_list)
else:
    build_net()
    video_scores = map(eval_video, eval_video_list)

video_pred = [np.argmax(default_aggregation_func(x[0])) for x in video_scores]
video_labels = [x[1] for x in video_scores]

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit/cls_cnt

print cls_acc

print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)

if save_scores is not None:
    np.savez(save_scores, scores=video_scores, labels=video_labels)
