import argparse
import sys
import numpy as np
import scipy.io as sio

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy

# parser = argparse.ArgumentParser()
# parser.add_argument('score_files', nargs='+', type=str)
# parser.add_argument('--score_weights', nargs='+', type=float, default=None)
# parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
# args = parser.parse_args()
# score_files = ["ucf101_split_1_rgb_seg124_3dropout_stage_2_lr_0.00001_iter_2100.npz", "ucf101_split_1_flow_seg124_3dropout_stage_2_lr_0.00001_iter_1500.npz"]
# score_files = ["ucf101_split_1_rgb_seg124_3dropout_imagenet_stage_2_lr_0.0001_iter_600.npz", "ucf101_split_1_flow_seg124_3dropout_stage_2_lr_0.00001_iter_1500.npz"]
# score_files = ["ucf101_split_1_rgb_seg124_3dropout_stage_2_lr_0.00001_iter_2100.npz", "ucf101_split_1_flow_seg124_dropout_0.9_lr_0.00001_iter_900.npz"]
#score_files = ["ucf101_split_2_rgb_seg124_3dropout_stage_2_imagenet_snapshot_lr_0.0001_iter_4500.npz", "ucf101_split_2_flow_seg124_3dropout_stage_2_lr_0.00001_iter_1200.npz"]
# score_files = ["ucf101_split_3_rgb_seg124_3dropout_imagenet_lr_0.0001_iter_600.npz", "ucf101_split_3_flow_seg124_3dropout_snapshot_stage_2_lr_0.0001_iter_900.npz"]
# score_files = ["ucf101_split_1_rgb_tpp_delete_dropout_lr_0.00001_iter_600.npz", "ucf101_split_1_flow_tpp_delete_dropout_lr_0.00001_iter_1500.npz"]
# score_files = ["hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112.npz", "hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672.npz"]

# score_files = ["hmdb51_split_2_rgb_tpp_delete_dropout_lr_0.00001_iter_112.npz", "hmdb51_split_2_flow_tpp_delete_dropout_lr_0.0001_iter_112.npz"]
# score_files = ["ucf101_split_2_rgb_tpp_delete_dropout_lr_0.0001_iter_1800.npz", "ucf101_split_2_flow_tpp_delete_dropout_sec_lr_0.00001_iter_1200.npz"]
# score_files = ["ucf101_split_3_rgb_tpp_delete_dropout_lr_0.0001_iter_900.npz", "ucf101_split_3_flow_tpp_delete_dropout_lr_0.00001_iter_1800.npz"]
# score_files = ["hmdb51_split_3_rgb_tpp_delete_dropout_lr_0.001_iter_2016.npz", "hmdb51_split_3_flow_tpp_delete_dropout_lr_0.0001_iter_224.npz"]
# score_files = ["ucf101_split_1_rgb_tpp_p1_lr_0.00001_iter_600.npz", "ucf101_split_1_flow_tpp_p1_lr_0.0001_iter_600.npz"]

# score_files = ["ucf101_split_1_rgb_tpp_delete_dropout_lr_0.00001_iter_600_varied_32.npz", "ucf101_split_1_flow_tpp_delete_dropout_lr_0.00001_iter_1500_varied_32.npz"]
# score_files = ["ucf101_split_1_rgb_tpp_p12_lr_0.00001_iter_1200.npz", "ucf101_split_1_flow_tpp_p12_lr_0.00001_iter_1200.npz"]

# score_files = ['ucf101_split_1_rgb_seg3_lr_0.00001_iter_300.npz', 'ucf101_split_1_flow_seg3_lr_0.0001_iter_600.npz']
# score_files = ['hmdb51_split_1_rgb_tpp_kinetics_lr_0.00001_iter_336.npz', 'hmdb51_split_1_flow_tpp_kinetics_lr_0.0001_iter_112.npz']
# score_files = ['hmdb51_split_2_rgb_tpp_kinetics_snapshot_lr_0.00001_iter_224.npz', 'hmdb51_split_2_flow_tpp_kinetics_lr_0.00001_iter_224.npz']
score_files = ['hmdb51_split_3_rgb_tpp_kinetics_lr_0.0001_iter_672.npz',
               'hmdb51_split_3_flow_tpp_kinetics_lr_0.00001_iter_224.npz']

# score_files = ['ucf101_split_1_rgb_tpp_p1248_lr_0.001_iter_600.npz', 'ucf101_split_1_flow_tpp_p1248_lr_0.00001_iter_1200.npz']
# score_files = ["ucf101_split_1_rgb_tpp_delete_dropout_lr_0.00001_iter_600.npz", 'ucf101_split_1_flow_tpp_imagenet_lr_0.00001_iter_900.npz']
# score_files = ['ucf101_split_1_rgb_tpp_p124_ave_snapshot_lr_0.00001_iter_600.npz', 'ucf101_split_1_flow_tpp_p124_ave_lr_0.00001_iter_600.npz']
# score_files = ["ucf101_split_1_rgb_tpp_freeze_cnn_lr_0.00001_iter_600.npz", 'ucf101_split_1_flow_tpp_freeze_cnn_lr_0.00001_iter_1200.npz']


# score_files = ['ucf101_split_1_rgb_tpp_kinetics_lr_0.0001_iter_1200.npz', 'ucf101_split_1_flow_tpp_kinetics_lr_0.0001_iter_600.npz'] ### 97.7%

score_files = ['ucf101_split_2_rgb_tpp_kinetics_lr_0.00001_iter_300.npz', 'ucf101_split_2_flow_tpp_kinetics_lr_0.0001_iter_600.npz']
score_files = ['ucf101_split_2_rgb_tpp_kinetics_st_0.001_lr_0.00001_iter_600.npz', 'ucf101_split_2_flow_tpp_kinetics_lr_0.0001_iter_600.npz']  ### 97.8 %
score_files = ['ucf101_split_1_rgb_tpp_kinetics_lr_0.0001_iter_1200.npz', 'ucf101_split_1_flow_tpp_kinetics_st_0.001_lr_0.0001_iter_900.npz']  ### 97.8%

score_files = ["hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112.npz", "hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672.npz"]

score_files = ['ucf101_split_2_rgb_tpp_kinetics_st_0.001_lr_0.00001_iter_600.npz', 'ucf101_split_2_flow_tpp_kinetics_st_0.001_lr_0.0001_iter_600.npz']

score_files = ['ucf101_split_1_rgb_tpp_seg_7_lr_0.00001_iter_300_varied_50.npz',
               'ucf101_split_1_flow_tpp_seg_7_lr_0.0001_iter_900_varied_50.npz']
score_files = ["ucf101_split_1_rgb_tpp_seg_4_lr_0.0001_iter_900_varied_50.npz",
"ucf101_split_1_flow_tpp_seg_4_lr_0.00001_iter_1200_varied_50.npz"]
score_files = ["ucf101_split_1_rgb_tpp_delete_dropout_lr_0.00001_iter_600_varied_50.npz",
             "ucf101_split_1_flow_tpp_delete_dropout_lr_0.00001_iter_1500_varied_50.npz"]


score_files = ['ucf101_split_1_rgb_tpp_kinetics_lr_0.0001_iter_1200.npz', 'ucf101_split_1_flow_tpp_kinetics_st_0.001_lr_0.0001_iter_900.npz']
score_files = ['ucf101_split_2_rgb_tpp_kinetics_st_0.001_lr_0.00001_iter_600.npz', 'ucf101_split_2_flow_tpp_kinetics_st_0.001_lr_0.0001_iter_600.npz']

score_files = ['ucf101_split_3_rgb_tpp_kinetics_lr_0.0001_iter_300.npz', 'ucf101_split_3_flow_tpp_kinetics_st_0.001_lr_0.00001_iter_300.npz']

score_files = ['ucf101_split_3_rgb_tpp_kinetics_st_0.001_lr_0.0001_iter_900.npz', 'ucf101_split_3_flow_tpp_kinetics_st_0.001_lr_0.00001_iter_300.npz']

score_files = ["hmdb51_split_1_rgb_tpp_delete_dropout_lr_0.0001_iter_112.npz", "hmdb51_split_1_flow_tpp_delete_dropout_lr_0.0001_iter_672.npz"]

score_files = ["hmdb51_split_1_tsn_rgb_reference_bn_inception_new.npz", "hmdb51_split_1_tsn_flow_reference_bn_inception_new.npz"]

score_files = ['ucf101_split_3_rgb_tpp_kinetics_st_0.01_lr_0.0001_iter_1500.npz', 'ucf101_split_3_flow_tpp_kinetics_st_0.001_lr_0.00001_iter_300.npz']


# save_scores = 'ucf101_split_2_rgb_tpp_kinetics_lr_0.00001_iter_300' ### 92.5%
# save_scores = 'ucf101_split_2_flow_tpp_kinetics_lr_0.0001_iter_600' ### 96.7%
# save_scores =  ### 93.81%


# score_files = ['ucf101_split_3_rgb_tpp_kinetics_lr_0.0001_iter_300.npz', 'ucf101_split_3_flow_tpp_kinetics_lr_0.0001_iter_1800.npz']

crop_agg = "mean"
xxxx = 0.5

score_npz_files = [np.load(x) for x in score_files]
score_list = [x['scores'][:, 0] for x in score_npz_files]
label_list = [x['labels'] for x in score_npz_files]

# label verification
# score_aggregation
agg_score_list = []
for score_vec in score_list:
    agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, crop_agg)) for x in score_vec]
    agg_score_list.append(np.array(agg_score_vec))
split = score_files[0].split("_")[2]

score_weights = [xxxx, 1.0 - xxxx]
# #
# if score_weights is None:
#     score_weights = [1] * len(score_npz_files)
# else:
#     score_weights = score_weights
#     if len(score_weights) != len(score_npz_files):
#         raise ValueError("Only {} weight specifed for a total of {} score files"
#                          .format(len(score_weights), len(score_npz_files)))
#
# final_scores = np.zeros_like(agg_score_list[0])
# for i, agg_score in enumerate(agg_score_list):
#     final_scores += agg_score * score_weights[i]
#
# print "split: ", split
# # accuracy
# # for x in final_scores:
# #     xx = x[0]
# #     xxx = xx[0]
# ff = [x[0][0] for x in final_scores]
# acc, class_acc = mean_class_accuracy(ff, label_list[0])
# print 'Final accuracy {:02f}%'.format(acc * 100)
# print "rgb_score_weight: ", xxxx
# print class_acc
# print "\n"



# deep temporal pyramid pooling
## only network prediction
for ii in xrange(0,11):
    xxxx = ii * 1.0 /10
    score_weights = [xxxx, 1.0-xxxx]

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
    # accuracy
    # for x in final_scores:
    #     xx = x[0]
    #     xxx = xx[0]
    ff = [x[0][0] for x in final_scores]
    acc, class_acc = mean_class_accuracy(ff, label_list[0])
    print 'Final accuracy {:02f}%'.format(acc * 100)
    print "rgb_score_weight: ", xxxx
    print "\n"


# MIFS fusion with our method ####
# xxxx = 0.4
# score_weights = [xxxx, 1.0-xxxx]
# if score_weights is None:
#     score_weights = [1] * len(score_npz_files)
# else:
#     score_weights = score_weights
#     if len(score_weights) != len(score_npz_files):
#         raise ValueError("Only {} weight specifed for a total of {} score files"
#                          .format(len(score_weights), len(score_npz_files)))
#
# final_scores = np.zeros_like(agg_score_list[0])
# for i, agg_score in enumerate(agg_score_list):
#     final_scores += agg_score * score_weights[i]
#
# print "split: ", split
# ff = [x[0][0] for x in final_scores]
# acc = mean_class_accuracy(ff, label_list[0])
# print 'Final accuracy {:02f}%'.format(acc * 100)
# print "rgb_score_weight: ", xxxx
# print "\n"
#
# test_score = "test_score_" + split
# matfn="MIFS_scores/hmdb/" + test_score
# data=sio.loadmat(matfn)
# MIFS_score = np.array(data[test_score])
# MIFS_score = MIFS_score.transpose(1,0)
# MIFS_score = MIFS_score.reshape(-1,1,1,51)
# for i in xrange(0,11):
#     MIFS_score_weight = i * 1.0 / 10
#     final_scores = (1-MIFS_score_weight) * final_scores +  MIFS_score_weight * MIFS_score
#     # accuracy
#     # for x in final_scores:
#     #     xx = x[0]
#     #     xxx = xx[0]
#     ff = [x[0][0] for x in final_scores]
#     # ff = final_scores
#     acc = mean_class_accuracy(ff, label_list[0])
#
#     print 'Final accuracy {:02f}%'.format(acc * 100)
#     print "MIFS_score_weight: ", MIFS_score_weight
#     print "\n"


# iDT fusion with our method ####
#
# xxxx = 0.5
# score_weights = [xxxx, 1.0 - xxxx]
# if score_weights is None:
#     score_weights = [1] * len(score_npz_files)
# else:
#     score_weights = score_weights
#     if len(score_weights) != len(score_npz_files):
#         raise ValueError("Only {} weight specifed for a total of {} score files"
#                          .format(len(score_weights), len(score_npz_files)))
#
# final_scores = np.zeros_like(agg_score_list[0])
# for i, agg_score in enumerate(agg_score_list):
#     final_scores += agg_score * score_weights[i]
#
# test_score = "idt_hmdb_test_score_" + split
# matfn= "iDT_scores/" + test_score + ".mat"
# data=sio.loadmat(matfn)
# iDT_score = np.array(data[test_score])
# iDT_score = iDT_score.transpose(1,0)
# iDT_score = iDT_score.reshape(-1,1,1,51)
#
# for i in xrange(0,11):
#     iDT_score_weight = i * 1.0 / 10
#     final_scores = (1-iDT_score_weight) * final_scores +  iDT_score_weight * iDT_score
#     # accuracy
#     # for x in final_scores:
#     #     xx = x[0]
#     #     xxx = xx[0]
#     ff = [x[0][0] for x in final_scores]
#     # ff = final_scores
#     acc = mean_class_accuracy(ff, label_list[0])
#     print 'Final accuracy {:02f}%'.format(acc * 100)
#     print "iDT_score_weight: ", iDT_score_weight
#     print '\n'