import argparse
import sys
import numpy as np
import scipy.io as sio

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy, class_accuracy


def get_score(score_files, xxxx = 0.4):
    crop_agg = "mean"
    score_npz_files = [np.load(x) for x in score_files]
    score_list = [x['scores'][:, 0] for x in score_npz_files]
    label_list = [x['labels'] for x in score_npz_files]
    agg_score_list = []
    for score_vec in score_list:
        agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, crop_agg)) for x in
                         score_vec]
        agg_score_list.append(np.array(agg_score_vec))
    split = score_files[0].split("_")[2]

    score_weights = [xxxx, 1.0 - xxxx]

    if score_weights is None:
        score_weights = [1] * len(score_npz_files)
    else:
        score_weights = score_weights
        if len(score_weights) != len(score_npz_files):
            raise ValueError("Only {} weight specifed for a total of {} score files"
                             .format(len(score_weights), len(score_npz_files)))

    final_scores = np.zeros_like(agg_score_list[0])
    for i, agg_score in enumerate(agg_score_list):
        final_scores += agg_score * score_weights[i]
    print "split: ", split
    ff = [x[0][0] for x in final_scores]
    return ff, label_list[0]
def get_score_11111(score_files, xxxx = 0.4):
    crop_agg = "mean"
    score_npz_files = [np.load(x) for x in score_files]
    score_list = [x['scores'][:, 0] for x in score_npz_files]
    label_list = [x['labels'] for x in score_npz_files]
    agg_score_list = []
    for score_vec in score_list:
        agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, crop_agg)) for x in
                         score_vec]
        agg_score_list.append(np.array(agg_score_vec))
    split = score_files[0].split("_")[2]

    score_weights = [xxxx, 1.0 - xxxx]

    if score_weights is None:
        score_weights = [1] * len(score_npz_files)
    else:
        score_weights = score_weights
        if len(score_weights) != len(score_npz_files):
            raise ValueError("Only {} weight specifed for a total of {} score files"
                             .format(len(score_weights), len(score_npz_files)))

    final_scores = np.zeros_like(agg_score_list[0])
    for i, agg_score in enumerate(agg_score_list):
        final_scores += agg_score * score_weights[i]
    print "split: ", split
    ff = [x[0][0] for x in final_scores]

    acc = mean_class_accuracy(ff, label_list[0])
    # print 'Final accuracy {:02f}%'.format(acc * 100)
    # print "rgb_score_weight: ", xxxx
    # print "\n"
    return acc


#
# # score_files = ['hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112_new_score.npz', 'hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672_new_score.npz']  ### 0.86
# score_files = ['hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112_swing_baseball.npz', 'hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672_swing_baseball.npz']
# score_files = ['hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112_draw_sword.npz', 'hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672_draw_sword.npz']
score_files = ['hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112_cart_wheel.npz', 'hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672_cart_wheel.npz']
ff, ll = get_score(score_files)
#
# # score_files = ['hmdb51_split_1_tsn_flow_reference_bn_inception_new_sit.npz', 'hmdb51_split_1_tsn_rgb_reference_bn_inception_new_sit.npz']
# score_files = ['hmdb51_split_1_tsn_rgb_reference_bn_inception_new_swing_baseball.npz', 'hmdb51_split_1_tsn_flow_reference_bn_inception_new_swing_baseball.npz']
# score_files = ['hmdb51_split_1_tsn_rgb_reference_bn_inception_new_draw_sword.npz', 'hmdb51_split_1_tsn_flow_reference_bn_inception_new_draw_sword.npz']
score_files = ['hmdb51_split_1_tsn_flow_reference_bn_inception_new_cart_wheel.npz', 'hmdb51_split_1_tsn_rgb_reference_bn_inception_new_cart_wheel.npz']
ff1, ll = get_score(score_files, xxxx=0.5)
#
acc, iiiii = class_accuracy(ff, ff1, ll)
# print 'Final accuracy {:02f}%'.format(acc * 100)
# print "rgb_score_weight: ", xxxx
# print "\n"



# score_files = ["hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112.npz", "hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672.npz"]
# score_files = ['hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112_swing_baseball.npz', 'hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672_swing_baseball.npz']
# score_files = ['hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112_draw_sword.npz', 'hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672_draw_sword.npz']
# score_files = ['hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112_cart_wheel.npz', 'hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672_cart_wheel.npz']
# acc_1 = get_score_11111(score_files)
# score_files = ["hmdb51_split_1_tsn_rgb_reference_bn_inception_new.npz", "hmdb51_split_1_tsn_flow_reference_bn_inception_new.npz"]
# score_files = ['hmdb51_split_1_tsn_rgb_reference_bn_inception_new_swing_baseball.npz', 'hmdb51_split_1_tsn_flow_reference_bn_inception_new_swing_baseball.npz']
# score_files = ['hmdb51_split_1_tsn_rgb_reference_bn_inception_new_draw_sword.npz', 'hmdb51_split_1_tsn_flow_reference_bn_inception_new_draw_sword.npz']
score_files = ['hmdb51_split_1_tsn_flow_reference_bn_inception_new_cart_wheel.npz', 'hmdb51_split_1_tsn_rgb_reference_bn_inception_new_cart_wheel.npz']
acc_2 = get_score_11111(score_files, xxxx=0.5)
#
# dev = acc_1 - acc_2
#
#
# def rr(nn1):
#     res = np.argsort(-nn1)
#     seq = nn1[res]
#     return res, seq
#
# ax, axx = rr(dev)
#
# print dev