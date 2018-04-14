"""
This module provides our implementation of different functions to do video-level classification and stream fusion
"""
import numpy as np
from metrics import softmax


def default_aggregation_func(score_arr, normalization=False, crop_agg=None):
    """
    This is the default function for make video-level prediction
    :param score_arr: a 3-dim array with (frame, crop, class) layout
    :return:
    """
    crop_agg = np.mean if crop_agg is None else crop_agg
    if normalization:
        a = softmax(crop_agg(score_arr, axis=1).mean(axis=0))
        return a
    else:
        a = crop_agg(score_arr, axis=1)
        a = a.mean(axis=0)
        a = a.reshape(1,1,-1)
        return a

def ave_aggregation_func(score_arr, score_arr_1, weight = 1, normalization=True, crop_agg=None):
    """
    This is the default function for make video-level prediction
    :param score_arr: a 3-dim array with (frame, crop, class) layout
    :return:
    """
    score_arr += weight * score_arr_1

    crop_agg = np.mean if crop_agg is None else crop_agg
    if normalization:
        return softmax(crop_agg(score_arr, axis=1).mean(axis=0))
    else:
        return crop_agg(score_arr, axis=1).mean(axis=0)

def sci_index(x, normalization = True):
    if normalization:
        x = softmax(x)
    scip = (101 * np.max(x)/np.sum(x) - 1)/(101 - 1)
    return scip

def sci_crop_aggregation_func(score_arr, score_arr_1, normalization=True, crop_agg=None):

    a,b,c = score_arr.shape[0:3]
    frame_score = []
    frame_score_1 = []
    for i in range(a):
        stream_sum = 0
        stream_sum_1 = 0
        scipp_sum = 0
        scipp_sum_1 = 0

        for j in range(b):
            stream = score_arr[i][j]
            stream_1 = score_arr_1[i][j]
            stream_sum += sci_index(stream)
            stream_sum_1 += sci_index(stream_1)
            scipp_sum += sci_index(stream) * softmax(stream)
            scipp_sum_1 += sci_index(stream) * softmax(stream_1)
        p1 = scipp_sum / stream_sum
        p2 = scipp_sum_1 / stream_sum_1

       # p = sci_index(p1) * softmax(p1) + sci_index(p2) * softmax(p2)/(sci_index(p1) + sci_index(p2))
        frame_score.append(p1)
        frame_score_1.append(p2)

    frame_score = np.array(frame_score)
    frame_score_1 = np.array(frame_score_1)

    frame_score = (frame_score).max(axis=0)
    frame_score_1 = (frame_score_1).max(axis=0)

    frame_score = softmax(frame_score)
    frame_score_1 = softmax(frame_score_1)

    frame_score += frame_score_1
    if normalization:
        return softmax((frame_score) )
    else:
        return (frame_score)

def sci_stream_aggregation_func(score_arr, score_arr_1, normalization=True, crop_agg=None):

    a,b,c = score_arr.shape[0:3]
    frame_score = []
    frame_score_1 = []
    for i in range(a):
        stream_sum = 0
        stream_sum_1 = 0
        scipp_sum = 0
        scipp_sum_1 = 0

        for j in range(b):
            stream = score_arr[i][j]
            stream_1 = score_arr_1[i][j]
            stream_sum += stream
            stream_sum_1 += stream_1
        p1 = stream_sum
        p2 = stream_sum_1

       # p = sci_index(p1) * softmax(p1) + sci_index(p2) * softmax(p2)/(sci_index(p1) + sci_index(p2))
        frame_score.append(p1)
        frame_score_1.append(p2)

    frame_score = np.array(frame_score)
    frame_score_1 = np.array(frame_score_1)

    frame_score = (frame_score).mean(axis=0)
    frame_score_1 = (frame_score_1).mean(axis=0)

    frame_score = softmax(frame_score)
    frame_score_1 = softmax(frame_score_1)

    frame_score = sci_index(frame_score, normalization=False) * frame_score + sci_index(frame_score_1, normalization=False) * frame_score_1

    # frame_score += frame_score_1
    #frame_score = (frame_score).max(axis=0)
    if normalization:
        return softmax((frame_score) )
    else:
        return (frame_score)



def sci_crop_stream_aggregation_func(score_arr, score_arr_1, normalization=True, crop_agg=None):

    a,b,c = score_arr.shape[0:3]
    frame_score = []
    frame_score_1 = []
    for i in range(a):
        stream_sum = 0
        stream_sum_1 = 0
        scipp_sum = 0
        scipp_sum_1 = 0

        for j in range(b):
            stream = score_arr[i][j]
            stream_1 = score_arr_1[i][j]
            stream_sum += sci_index(stream)
            stream_sum_1 += sci_index(stream_1)
            scipp_sum += sci_index(stream) * softmax(stream)
            scipp_sum_1 += sci_index(stream) * softmax(stream_1)
        p1 = scipp_sum / stream_sum
        p2 = scipp_sum_1 / stream_sum_1

       # p = sci_index(p1) * softmax(p1) + sci_index(p2) * softmax(p2)/(sci_index(p1) + sci_index(p2))
        frame_score.append(p1)
        frame_score_1.append(p2)

    frame_score = np.array(frame_score)
    frame_score_1 = np.array(frame_score_1)

    frame_score = (frame_score).max(axis=0)
    frame_score_1 = (frame_score_1).max(axis=0)

    frame_score = softmax(frame_score)
    frame_score_1 = softmax(frame_score_1)

    frame_score = sci_index(frame_score, normalization=False) * frame_score + sci_index(frame_score_1, normalization=False) * frame_score_1

    # frame_score += frame_score_1
    #frame_score = (frame_score).max(axis=0)
    if normalization:
        return softmax((frame_score) )
    else:
        return (frame_score)


def sliding_window_aggregation_func(score, spans=[1, 2, 4, 8, 16], overlap=0.2, norm=True, fps=1):
    """
    This is the aggregation function used for ActivityNet Challenge 2016
    :param score:
    :param spans:
    :param overlap:
    :param norm:
    :param fps:
    :return:
    """
    frm_max = score.max(axis=1)
    slide_score = []

    def top_k_pool(scores, k):
        return np.sort(scores, axis=0)[-k:, :].mean(axis=0)

    for t_span in spans:
        span = t_span * fps
        step = int(np.ceil(span * (1-overlap)))
        local_agg = [frm_max[i: i+span].max(axis=0) for i in xrange(0, frm_max.shape[0], step)]
        k = max(15, len(local_agg)/4)
        slide_score.append(top_k_pool(np.array(local_agg), k))

    out_score = np.mean(slide_score, axis=0)

    if norm:
        return softmax(out_score)
    else:
        return out_score


def default_fusion_func(major_score, other_scores, fusion_weights, norm=True):
    assert len(other_scores) == len(fusion_weights)
    out_score = major_score
    for s, w in zip(other_scores, fusion_weights):
        out_score += s * w

    if norm:
        return softmax(out_score)
    else:
        return out_score
