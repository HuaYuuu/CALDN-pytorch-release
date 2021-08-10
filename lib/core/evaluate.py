# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize, target_weight):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            # if target[n, c, 0] > 1 and target[n, c, 1] > 1:
            if target_weight is not None:
                if target_weight[n, c] > 0.5:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
            else:
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def dist_acc_withsum(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal, \
               int(np.less(dists[dist_cal], thr).sum() * 1.0), num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, target_weight = get_max_preds(target)  # target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm, target_weight)  # dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def accuracy_withclass(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''

    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, target_weight = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm, target_weight)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    dist_sum = dists[dist_cal].sum()
    nme = dist_sum / num_dist_cal

    class_sum = np.zeros_like(pred)[0, :, 0]
    class_right = np.zeros_like(pred)[0, :, 0]

    for i in range(len(idx)):
        acc[i + 1], temp_right, temp_sum = dist_acc_withsum(dists[idx[i]])
        class_sum[i] = temp_sum
        class_right[i] = temp_right
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred, class_sum, class_right, dist_sum, num_dist_cal


def accuracy_withclassandrate(output, target, hm_type='gaussian', thr=0.5, wh_rate=None):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''

    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, target_weight = get_max_preds(target)

        norms = []
        for index in range(output.shape[0]):
            if wh_rate is not None:
                assert len(wh_rate) == output.shape[0]
                if wh_rate[index] >= 1:
                    w = output.shape[3]
                    h = int(w / wh_rate[index])
                else:
                    h = output.shape[2]
                    w = int(h * wh_rate[index])
            else:
                h = output.shape[2]
                w = output.shape[3]
            norm = np.ones((1, 2)) * np.array([h, w]) / 10
            norms.append(norm)
        norm = np.array(norms)
        norm = np.squeeze(norm, axis=1)

    dists = calc_dists_withrate(pred, target, norm, target_weight)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    class_sum = np.zeros_like(pred)[0, :, 0]
    class_right = np.zeros_like(pred)[0, :, 0]

    for i in range(len(idx)):
        acc[i + 1], temp_right, temp_sum = dist_acc_withsum(dists[idx[i]])
        class_sum[i] = temp_sum
        class_right[i] = temp_right
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred, class_sum, class_right


def accuracy_savewrongpoints(output, target, hm_type='gaussian', thr=0.5):

    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, target_weight = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm, target_weight)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    right_wrong_matrix = np.zeros_like(dists)

    for i in range(len(idx)):
        right_points = dist_acc_withoutsum(dists[idx[i]])
        if right_points is not None:
            right_wrong_matrix[i] = right_points

    return right_wrong_matrix


def dist_acc_withoutsum(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        distance_without_invisible = dist_cal * dists
        right_points = np.less(distance_without_invisible, thr)
        return right_points
    else:
        return None


def calc_dists_withrate(preds, target, normalize, target_weight):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            # if target[n, c, 0] > 1 and target[n, c, 1] > 1:
            if target_weight is not None:
                if target_weight[n, c] > 0.5:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
            else:
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
    return dists
