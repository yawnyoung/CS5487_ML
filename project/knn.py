"""
KNN classifier

author: Yajue Yang
"""

from project.data_processing import load_train_test_data
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

    for i in range(len(test_feat)):

        e_tt = classify(dist_func, k, train_feat, train_tgt, test_feat[i])

        print(test_tgt[i], e_tt)


if __name__ == '__main__':
    (train_tgt, train_feat), (test_tgt, test_feat)= load_train_test_data()

    test_stat(euclidean_dist, 3, train_feat, train_tgt, test_feat, test_tgt)
