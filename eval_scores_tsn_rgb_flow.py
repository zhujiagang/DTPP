import argparse
import sys
import numpy as np
sys.path.append('.')

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy

score_files = ["hmdb51_split_1_tsn_rgb_score.npz", "hmdb51_split_1_tsn_flow_reference_bn_inception.npz"]

crop_agg = "mean"
xxxx = 1
score_weights = [xxxx, 1.0-xxxx]
score_npz_files = [np.load(x) for x in score_files]


if score_weights is None:
    score_weights = [1] * len(score_npz_files)
else:
    score_weights = score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))

score_list = [x['scores'][:, 0] for x in score_npz_files]
label_list = [x['labels'] for x in score_npz_files]

# label verification

# score_aggregation
agg_score_list = []
for score_vec in score_list:
    agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, crop_agg)) for x in score_vec]
    agg_score_list.append(np.array(agg_score_vec))

final_scores = np.zeros_like(agg_score_list[0])
for i, agg_score in enumerate(agg_score_list):
    final_scores += agg_score * score_weights[i]

# accuracy
acc = mean_class_accuracy(final_scores, label_list[0])
print 'Final accuracy {:02f}%'.format(acc * 100)