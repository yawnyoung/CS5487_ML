"""
Linear discriminant algorithm with LS minimization

author: Yajue Yang
"""

import numpy as np
from project.data_processing import load_train_test_data, add_gaussian_noises
from project.utils import *


def tgt_binary_encode(t):
    """
    1-of-K coding scheme
    :param t: belong to {1, 2, 3}
    :return:
    """
    if t == 1:
        return np.array([1, 0, 0])
    elif t == 2:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


def decision_rule(t):
    """
    decision rule: the class corresponding to the maximum value
    :param t:
    :return:
    """
    return np.argmax(t) + 1


def get_data_matrix(tgt_feature_pair):
    """
    get data in form of matrix to facilitate calculations
    :param tgt_feature_pair: pair of targets and features
    :return:
    """
    tgt = tgt_feature_pair[0]
    feat = tgt_feature_pair[1]
    assert len(tgt) == len(feat)

    num_var = len(feat[0])

    tgt_arr = np.ndarray(shape=(len(tgt), 3), dtype=int)
    feat_arr = np.ones(shape=(len(feat), num_var + 1), dtype=float)

    for i in range(len(tgt)):
        tgt_arr[i] = tgt_binary_encode(tgt[i])
        feat_arr[i][1:] = np.array(feat[i], dtype=float)

    return tgt_arr, feat_arr


def train_w(train_data_mat):
    """
    calculate parameter w from training data
    :param train_data_mat: pair of training target and feature matrices
    :return:
    """
    y = train_data_mat[0]
    x = train_data_mat[1]
    x_t = np.transpose(x)

    w = np.linalg.inv(x_t @ x) @ x_t @ y

    return w


def classify_data(w, feat):

    w_t = np.transpose(w)

    y_classified = w_t @ feat

    return decision_rule(y_classified)


def test_stat(w, test_data):

    test_targets = test_data[0]
    test_data_mat = get_data_matrix(test_data)
    test_feat_arr = test_data_mat[1]

    num_test = len(test_targets)

    accuracy = 0

    for i in range(len(test_targets)):
        c = classify_data(w, test_feat_arr[i])

        if c == test_targets[i]:
            accuracy += 1

    return accuracy / num_test


def accuracy_test(t_size, add_noise=False, sigma_sqr=None, ret_acc=True):

    acc = 0

    for i in range(NUM_RANDOM):

        train_data, test_data = load_train_test_data(t_size, True)

        if add_noise:
            train_data = add_gaussian_noises(train_data, sigma_sqr)

        train_data_mat = get_data_matrix(train_data)

        w = train_w(train_data_mat)

        acc += test_stat(w, test_data)

    acc /= NUM_RANDOM

    if ret_acc:
        return acc
    else:
        return 1 - acc


def accuracy_against_size_stat(train_sizes, ret_acc=True):

    acc = []

    for s in train_sizes:
        acc.append(accuracy_test(s, ret_acc=ret_acc))

    return acc


def accuracy_against_noise_stat(sigma_sqrs, train_size=0.9, ret_acc=True):

    acc = []

    for s in sigma_sqrs:
        print('sigma square: ', s)
        acc.append(accuracy_test(train_size, add_noise=True, sigma_sqr=s, ret_acc=ret_acc))

    return acc


if __name__ == '__main__':

    train_data, test_data = load_train_test_data(0.9)

    train_data_mat = get_data_matrix(train_data)

    w = train_w(train_data_mat)

    test_stat(w, test_data)

