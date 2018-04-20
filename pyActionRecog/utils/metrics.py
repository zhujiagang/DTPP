"""
This module provides some utils for calculating metrics
"""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]


def top_k_acc(lb_set, scores, k=3):
    idx = np.argsort(scores)[-k:]
    return len(lb_set.intersection(idx)), len(lb_set)


def top_k_hit(lb_set, scores, k=3):
    idx = np.argsort(scores)[-k:]
    return len(lb_set.intersection(idx)) > 0, 1


def top_3_accuracy(score_dict, video_list):
    return top_k_accuracy(score_dict, video_list, 3)


def top_k_accuracy(score_dict, video_list, k):
    video_labels = [set([i.num_label for i in v.instances]) for v in video_list]

    video_top_k_acc = np.array(
        [top_k_hit(lb, score_dict[v.id], k=k) for v, lb in zip(video_list, video_labels)
         if v.id in score_dict])

    tmp = video_top_k_acc.sum(axis=0).astype(float)
    top_k_acc = tmp[0] / tmp[1]

    return top_k_acc


def video_mean_ap(score_dict, video_list):
    avail_video_labels = [set([i.num_label for i in v.instances]) for v in video_list if
                          v.id in score_dict]
    pred_array = np.array([score_dict[v.id] for v in video_list if v.id in score_dict])
    gt_array = np.zeros(pred_array.shape)

    for i in xrange(pred_array.shape[0]):
        gt_array[i, list(avail_video_labels[i])] = 1
    mean_ap = average_precision_score(gt_array, pred_array, average='macro')
    return mean_ap


def mean_class_accuracy(scores, labels):


    pred = np.argmax(scores, axis=1)

    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    return np.mean(cls_hit/cls_cnt), cls_acc
    # return


def class_accuracy(scores, scores_1, labels):
#     rgb = [x[0][0] for x in rgb]
#     flow = [x[0][0] for x in flow]

    def rr(aa,ii):
        nn1 = aa[ii]
        res = np.argsort(-nn1)
        seq = nn1[res]
        return res, seq
    ii = 6


    DTPP_1, DTPP_2 = rr(scores,ii)
    DTPP_score = DTPP_2


    TSN_1, TSN_2 = rr(scores_1, ii)
    TSN_score = TSN_2

    # rgb_1, rgb_2 = rr(rgb, ii)
    # flow_1, flow_2 = rr(flow, ii)

    pred_DTPP = np.argmax(scores, axis=1)

    pred_TSN = np.argmax(scores_1, axis=1)


    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    print cls_acc

    return np.mean(cls_hit/cls_cnt), cls_acc