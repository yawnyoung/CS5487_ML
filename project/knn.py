"""
KNN classifier

author: Yajue Yang
"""

from project.data_processing import load_train_test_data, add_gaussian_noises
from project.utils import *
import numpy as np


def get_k_neighbors(dist_func, k, train_feat, test_feat):
    """
    get k neighbors in the training dataset of a test instance
    :param dist_func: distance function
    :param k:
    :param train_data:
    :param test_feat:
    :return: indices
    """

    distances = []

    for t_f in train_feat:
        distances.append(dist_func(t_f, test_feat))

    idx = np.argsort(np.array(distances))

    return idx[:k]


def classify(dist_func, k, train_feat, train_tgt, test_feat):

    k_indices = get_k_neighbors(dist_func, k, train_feat, test_feat)

    num_tgt = [0, 0, 0]

    for i in k_indices:
        num_tgt[train_tgt[i] - 1] += 1

    return np.argmax(np.array(num_tgt)) + 1


def test_stat(dist_func, k, train_feat, train_tgt, test_feat, test_tgt):

    acc = 0

    for i in range(len(test_feat)):

        e_tt = classify(dist_func, k, train_feat, train_tgt, test_feat[i])

        if test_tgt[i] == e_tt:
            acc += 1

    return acc / len(test_feat)


def accuracy_against_size(t_size, add_noise=False, sigma_sqr=None, ret_acc=True):

    acc = 0

    for i in range(NUM_RANDOM):

        (train_tgt, train_feat), (test_tgt, test_feat) = load_train_test_data(t_size, True)

        if add_noise:
            (train_tgt, train_feat) = add_gaussian_noises((train_tgt, train_feat), sigma_sqr)

        acc += test_stat(euclidean_dist, 3, train_feat, train_tgt, test_feat, test_tgt)

    acc /= NUM_RANDOM

    if ret_acc:
        return acc
    else:
        return 1 - acc


def accuracy_against_size_stat(train_sizes, add_noise=False, sigma_sqr=None, ret_acc=True):

    acc = []

    for s in train_sizes:
        acc.append(accuracy_against_size(s, add_noise, sigma_sqr, ret_acc))

    return acc


if __name__ == '__main__':
    (train_tgt, train_feat), (test_tgt, test_feat)= load_train_test_data()

    test_stat(euclidean_dist, 3, train_feat, train_tgt, test_feat, test_tgt)
