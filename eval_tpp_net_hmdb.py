import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix


relative_path = "/home/lilin/my_code"
caffe_root = relative_path + '/deeptemporal/lib/caffe-tpp-net-python/'
sys.path.insert(0, caffe_root + 'python')

from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file

from pyActionRecog.utils.video_funcs import default_aggregation_func


# net_weights = "/home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112.caffemodel"

net_weights = "/home/lilin/my_code/hmdb51_split_1_rgb_flow_models/hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672.caffemodel"


memory_enough = False

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

net_proto = relative_path + "/deeptemporal/models/"+ dataset + "/" + modality + "_tpp_delete_dropout_deploy.prototxt"

rgb_prefix ='img_'
flow_x_prefix ='flow_x_'
flow_y_prefix ='flow_y_'
num_frame_per_video = 25
num_id = 1
num_worker = 1
num_scores = 1
gpus = [1]

print "dataset: ", dataset, "split: ", split, "modality: ", modality, "gpu: ", num_id

# save_scores = 'hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112_cart_wheel'
save_scores = 'hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672_cart_wheel'

print save_scores

from pyActionRecog.action_caffe import CaffeNet

# build neccessary information
print dataset
split_tp = parse_split_file('newhmdb51')
f_info = parse_directory(frame_path,
                         rgb_prefix, flow_x_prefix, flow_y_prefix)

gpu_list = gpus


eval_video_list = split_tp[split - 1][1]

score_name = 'pool_fusion'
global ii
ii = 0

def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if num_worker > 1 else 1
    print "my", my_id
    if num_worker == 1:
        net = CaffeNet(net_proto, net_weights, num_id)
    else:
        if gpu_list is None:
            net = CaffeNet(net_proto, net_weights, my_id-1)
        else:
            net = CaffeNet(net_proto, net_weights, gpu_list[my_id - 1])


def eval_video(video):
    global net
    label = video[1]
    vid = video[0]
    video_frame_path = f_info[0][vid]
    if modality == 'rgb':
        cnt_indexer = 1
    elif modality == 'flow':
        cnt_indexer = 2
    else:
        raise ValueError(modality)
    frame_cnt = f_info[cnt_indexer][vid]

    stack_depth = 0
    if modality == 'rgb':
        stack_depth = 1
    elif modality == 'flow':
        stack_depth = 5

    step = 1.0 * (frame_cnt - stack_depth) / (num_frame_per_video-1)
    if step > 0:
        frame_ticks = []
        for i in range(num_frame_per_video):
            frame_ticks.append(int(1 + i * step))
    else:
        frame_ticks = [1] * num_frame_per_video

    assert(len(frame_ticks) == num_frame_per_video)

    frame_scores = []

    if modality == 'rgb':
        rgb_stack = []
        for iii in range(num_frame_per_video):
            tick = frame_ticks[iii]
            name = '{}{:05d}.jpg'.format(rgb_prefix, tick)
            frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
            rgb_stack.append(frame)

        if memory_enough is False:
            scores = net.predict_single_rgb_stack_memory(rgb_stack, score_name, frame_size=(340, 256), stack_len=num_frame_per_video)
        else:
            scores = net.predict_single_rgb_stack(rgb_stack, score_name, frame_size=(340, 256),
                                                         stack_len=num_frame_per_video)
        frame_scores.append(scores)

    if modality == 'flow':
        flow_stack = []
        for iii in range(num_frame_per_video):
            tick = frame_ticks[iii]
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
            flow_one = np.empty((256, 340, 10), dtype=np.float32)
            ix = 0
            for idx in frame_idx:
                x_name = '{}{:05d}.jpg'.format(flow_x_prefix, idx)
                y_name = '{}{:05d}.jpg'.format(flow_y_prefix, idx)
                frame_1 = cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE)
                frame_2 = cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE)
                flow_one[:,:,ix] = frame_1
                ix += 1
                flow_one[:,:,ix] = frame_2
                ix += 1
            flow_stack.append(flow_one)
        if memory_enough is False:
            scores = net.predict_single_flow_stack_test_memory(flow_stack, score_name, frame_size=(340, 256), stack_len=num_frame_per_video)
        else:
            scores = net.predict_single_flow_stack_test(flow_stack, score_name, frame_size=(340, 256))

        frame_scores.append(scores)
    global ii
    ii += 1
    print ii, 'video {} done'.format(vid)
    sys.stdin.flush()
    print "label: ", label
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