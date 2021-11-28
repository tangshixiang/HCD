
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import os
import torch
import numpy as np
import glob
from sklearn.metrics.cluster import (contingency_matrix,
                                     normalized_mutual_info_score)
from sklearn.metrics import (precision_score, recall_score)

__all__ = ['pairwise', 'bcubed', 'nmi', 'precision', 'recall', 'accuracy']


def _check(gt_labels, pred_labels):
    if gt_labels.ndim != 1:
        raise ValueError("gt_labels must be 1D: shape is %r" %
                         (gt_labels.shape, ))
    if pred_labels.ndim != 1:
        raise ValueError("pred_labels must be 1D: shape is %r" %
                         (pred_labels.shape, ))
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            "gt_labels and pred_labels must have same size, got %d and %d" %
            (gt_labels.shape[0], pred_labels.shape[0]))
    return gt_labels, pred_labels


def _get_lb2idxs(labels):
    lb2idxs = {}
    for idx, lb in enumerate(labels):
        if lb not in lb2idxs:
            lb2idxs[lb] = []
        lb2idxs[lb].append(idx)
    return lb2idxs


def _compute_fscore(pre, rec):
    return 2. * pre * rec / (pre + rec)


def fowlkes_mallows_score(gt_labels, pred_labels, sparse=True):
    ''' The original function is from `sklearn.metrics.fowlkes_mallows_score`.
        We output the pairwise precision, pairwise recall and F-measure,
        instead of calculating the geometry mean of precision and recall.
    '''
    n_samples, = gt_labels.shape

    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel()**2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel()**2) - n_samples

    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def pairwise(gt_labels, pred_labels, sparse=True):
    _check(gt_labels, pred_labels)
    return fowlkes_mallows_score(gt_labels, pred_labels, sparse)


def bcubed(gt_labels, pred_labels):
    _check(gt_labels, pred_labels)

    gt_lb2idxs = _get_lb2idxs(gt_labels)
    pred_lb2idxs = _get_lb2idxs(pred_labels)

    num_lbs = len(gt_lb2idxs)
    pre = np.zeros(num_lbs)
    rec = np.zeros(num_lbs)
    gt_num = np.zeros(num_lbs)

    for i, gt_idxs in enumerate(gt_lb2idxs.values()):
        all_pred_lbs = np.unique(pred_labels[gt_idxs])
        gt_num[i] = len(gt_idxs)
        for pred_lb in all_pred_lbs:
            pred_idxs = pred_lb2idxs[pred_lb]
            n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
            pre[i] += n**2 / len(pred_idxs)
            rec[i] += n**2 / gt_num[i]

    gt_num = gt_num.sum()
    avg_pre = pre.sum() / gt_num
    avg_rec = rec.sum() / gt_num
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def nmi(gt_labels, pred_labels):
    return normalized_mutual_info_score(pred_labels, gt_labels)


def precision(gt_labels, pred_labels):
    return precision_score(gt_labels, pred_labels)


def recall(gt_labels, pred_labels):
    return recall_score(gt_labels, pred_labels)


def accuracy(gt_labels, pred_labels):
    return np.mean(gt_labels == pred_labels)

if __name__=="__main__":
    dataset_name_1 = 'market1501'
    label_dir_1 = 'ablation_logs/21_306_final_iter_5_uda_trainer_update_method_4_new_method17_con_30_thre0.45_add_split_self_split_method10_alpha_0.99_1.0_nei_64_0.25_0.25_0.25_gap_1_split_8_d_to_m'
    dataset_name_2 = 'dukemtmc'
    label_dir_2 = 'ablation_logs/21_303_final_iter_5_uda_trainer_update_method_4_new_method15_con_64_thre0.45_add_split_self_split_method10_alpha_0.5_wo_outlier_0_1.0_nei_64_0.25_0.25_0.25_split_8_m_to_duke'

    ######market########
    print('-------------market----------------')
    all_label = glob.glob(os.path.join(label_dir_1, 'label_*.pth.tar'))
    all_label.sort()
    gt_label_tar = 'true_lab/{}.pth.tar'.format(dataset_name_1)
    gt_label_npy = 'true_lab/{}.npy'.format(dataset_name_1)
    gt_label = np.array(torch.load(gt_label_tar).tolist())
    labels = []
    pre_1, rec_1 = [], []
    for idx, ii in enumerate(all_label):
        layer_lab = np.array(torch.load(ii).tolist())
        labels.append(layer_lab)

    for idx, ii in enumerate(all_label):
        print('idx:', idx)
        # import pdb;pdb.set_trace()
        assert ii.split('/')[-1].startswith('label_{}'.format(idx))
        layer_lab = np.array(torch.load(ii).tolist())
        # rec_i= recall(gt_label, layer_lab)
        # pre_i= precision(gt_label,layer_lab)
        # pre_1.append(pre_i)
        # rec_1.append(rec_i)

        pair_f=pairwise(gt_label,layer_lab)
        bcub_f=bcubed(gt_label,layer_lab)
        print('pair_f:{}\nbcube_f:{}'.format(pair_f,bcub_f))

    ######duke########
    print('-------------duke-------------')
    all_label = glob.glob(os.path.join(label_dir_2, 'label_*.pth.tar'))
    all_label.sort()
    gt_label_tar = 'true_lab/{}.pth.tar'.format(dataset_name_2)
    gt_label = np.array(torch.load(gt_label_tar).tolist())
    labels = []
    pre_2, rec_2 = [], []
    for idx, ii in enumerate(all_label):
        layer_lab = np.array(torch.load(ii).tolist())
        labels.append(layer_lab)

    for idx, ii in enumerate(all_label):
        print('idx:', idx)
        # import pdb;pdb.set_trace()
        assert ii.split('/')[-1].startswith('label_{}'.format(idx))
        layer_lab = np.array(torch.load(ii).tolist())
        # rec_i = recall(gt_label, layer_lab)
        # pre_i = precision(gt_label, layer_lab)
        # pre_1.append(pre_i)
        # rec_1.append(rec_i)

        pair_f = pairwise(gt_label, layer_lab)
        bcub_f = bcubed(gt_label, layer_lab)
        print('pair_f:{}\nbcube_f:{}'.format(pair_f, bcub_f))