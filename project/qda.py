"""
Quadratic discriminant algorithm

author: Yajue Yang
"""

from project.data_processing import load_train_test_data, add_gaussian_noises
from project.lda_ls import tgt_binary_encode
import numpy as np
from project.utils import *
import matplotlib.pyplot as plt


def get_data_matrix(tgt_feature_pair):
    tgt = tgt_feature_pair[0]
    feat = tgt_feature_pair[1]
    assert len(tgt) == len(feat)

    num_var = len(feat[0])

    tgt_arr = np.ndarray(shape=(len(tgt), 3), dtype=int)
    feat_arr = np.ones(shape=(len(feat), num_var), dtype=float)

    for i in range(len(tgt)):
        tgt_arr[i] = tgt_binary_encode(tgt[i])
        feat_arr[i, :] = np.array(feat[i], dtype=float)

    return tgt_arr, feat_arr


def train_params(train_data_mat):

    y = train_data_mat[0]
    x = train_data_mat[1]

    num_data = x.shape[0]
    dim_param = x.shape[1]

    x_mean = np.zeros((3, dim_param))
    count_class = np.zeros(3)

    x_classified = [[], [], []]

    for i in range(num_data):
        class_idx = np.argmax(y[i])
        x_mean[class_idx, :] += x[i, :]
        count_class[class_idx] += 1
        x_classified[int(class_idx)].append(x[i, :])

    x_mean = np.divide(x_mean, count_class[:, np.newaxis])

    x_cov = np.zeros((3, dim_param, dim_param))

    # print(x_mean)

    for k in range(3):
        # print(x_mean[k])
        for x_c in x_classified[k]:
            # print(x_c)
            x_diff = (x_c - x_mean[k])[:, np.newaxis]
            # print(x_diff.T)
            x_cov[k, :, :] += x_diff @ x_diff.T

    # for k in range(3):
    #     print(x_cov[k, :, :])

    for k in range(3):
        x_cov[k, :, :] /= count_class[k]

    return x_mean, x_cov


def classify_data(x_mean, x_cov, feat):

    z = []

    for k in range(3):
        x_diff = (feat - x_mean[k, :])[:, np.newaxis]
        # print(np.linalg.det(x_cov[k, :, :]))
        score = -0.5 * x_diff.T @ np.linalg.inv(x_cov[k, :, :]) @ x_diff - 0.5 * np.log(np.linalg.det(x_cov[k, :, :]))
        # print(score)
        z.append(score)

    return np.argmax(np.array(z)) + 1


def test_stat(x_mean, x_cov, test_data):

    test_targets = test_data[0]

    num_test = len(test_targets)

    accuracy = 0

    for i in range(num_test):
        c = classify_data(x_mean, x_cov, np.array(test_data[1][i]))

        if c == test_targets[i]:
            accuracy += 1

    print('single accuracy: ', accuracy / num_test)

    return accuracy / num_test


def accuracy_test(t_size, add_noise=False, sigma_sqr=None, ret_acc=True):

    acc = 0

    for i in range(NUM_RANDOM):

        train_data, test_data = load_train_test_data(t_size, True)

        if add_noise:
            train_data = add_gaussian_noises(train_data, sigma_sqr)

        train_data_mat = get_data_matrix(train_data)

        x_mean, x_cov = train_params(train_data_mat)

        acc += test_stat(x_mean, x_cov, test_data)

    acc /= NUM_RANDOM

    print('all accuracy: ', acc)

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

    train_sizes = np.arange(0.1, 0.9, 10)
    accuracy_against_size_stat(train_sizes)