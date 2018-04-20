import argparse
import sys
import numpy as np

caffe_root = '/home/zhujiagang/temporal-segment-networks/lib/caffeaction/'
sys.path.insert(0, caffe_root + 'python')


#sys.path.append('.')

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy


#parser = argparse.ArgumentParser()
#parser.add_argument('score_files', nargs='+', type=str)
#parser.add_argument('--score_weights', nargs='+', type=float, default=None)
#parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
#args = parser.parse_args()

score_files = ['/home/zjg/zjg/tsncaffe/score/hmdb51_split_3_tsn_rgb_score.npz', '/home/zjg/zjg/tsncaffe/score/hmdb51_split_3_tsn_flow_score.npz']
score_npz_files = [np.load(x) for x in score_files]
score_list = [x['scores'][:, 0] for x in score_npz_files]
print len(score_list)
label_list = [x['labels'] for x in score_npz_files]
print len(label_list)
for ii in xrange(11):
    print ii
    score_weights = [ii,10-ii]

    if score_weights is None:
        score_weights = [1] * len(score_npz_files)
    else:
        score_weights = score_weights
        if len(score_weights) != len(score_npz_files):
            raise ValueError("Only {} weight specifed for a total of {} score files"
                             .format(len(score_weights), len(score_npz_files)))
    # label verification
    iii = 0
    # score_aggregation
    agg_score_list = []
    for score_vec in score_list:
        iii = iii + 1
        agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, 'mean')) for x in score_vec]
        #print len(agg_score_vec)
        agg_score_list.append(np.array(agg_score_vec))
    #print iii
    iii = 0
    final_scores = np.zeros_like(agg_score_list[0])
    for i, agg_score in enumerate(agg_score_list):
        iii = iii +1
        final_scores += agg_score * score_weights[i]

    #print iii
    #print len(label_list[0])
    #print len(final_scores)
    # accuracy
    acc = mean_class_accuracy(final_scores, label_list[0])
    print 'Final accuracy {:02f}%'.format(acc * 100)