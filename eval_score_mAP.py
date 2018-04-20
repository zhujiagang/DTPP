import argparse
import sys
import numpy as np

caffe_root = '/home/zhujiagang/temporal-segment-networks/lib/caffeaction/'
sys.path.insert(0, caffe_root + 'python')

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy

score_files = [['/home/zjg/zjg/tsncaffe/score/ucf101_split_1_tsn_rgb_score.npz', '/home/zjg/zjg/tsncaffe/score/ucf101_split_1_tsn_flow_score.npz'],
               ['/home/zjg/zjg/tsncaffe/score/ucf101_split_2_tsn_rgb_score.npz', '/home/zjg/zjg/tsncaffe/score/ucf101_split_2_tsn_flow_score.npz'],
               ['/home/zjg/zjg/tsncaffe/score/ucf101_split_3_tsn_rgb_score.npz', '/home/zjg/zjg/tsncaffe/score/ucf101_split_3_tsn_flow_score.npz']]

score_files = [['/home/zjg/zjg/tsncaffe/score/hmdb51_split_1_tsn_rgb_score.npz', '/home/zjg/zjg/tsncaffe/score/hmdb51_split_1_tsn_flow_score.npz'],
               ['/home/zjg/zjg/tsncaffe/score/hmdb51_split_2_tsn_rgb_score.npz', '/home/zjg/zjg/tsncaffe/score/hmdb51_split_2_tsn_flow_score.npz'],
               ['/home/zjg/zjg/tsncaffe/score/hmdb51_split_3_tsn_rgb_score.npz', '/home/zjg/zjg/tsncaffe/score/hmdb51_split_3_tsn_flow_score.npz']]
ii=3
accuracy = [[] for row in range(ii)]
for iii in xrange(ii):

    print 'ii', iii
    score_npz_files = [np.load(x) for x in score_files[iii]]
    score_list = [x['scores'][:, 0] for x in score_npz_files]
    label_list = [x['labels'] for x in score_npz_files]

    for ii in xrange(11):
        print ii
        score_weights = [ii,10-ii]
        # label verification
        # score_aggregation
        agg_score_list = []
        for score_vec in score_list:
            agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, 'mean')) for x in score_vec]
            #print len(agg_score_vec)
            agg_score_list.append(np.array(agg_score_vec))

        final_scores = np.zeros_like(agg_score_list[0])
        for i, agg_score in enumerate(agg_score_list):
            final_scores += agg_score * score_weights[i]
        # accuracy
        acc = mean_class_accuracy(final_scores, label_list[0])
        #print acc
        print 'Final accuracy {:02f}%'.format(acc * 100)
        accuracy[iii].append(np.array(acc))
        #print accuracy


print '\nFinal mAP:'
print np.mean(accuracy, axis = 0)