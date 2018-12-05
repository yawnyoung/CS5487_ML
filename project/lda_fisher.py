"""
Fisher linear discriminant algorithm

author: Yajue Yang
"""

from project.data_processing import load_train_test_data, add_gaussian_noises
from project.utils import *
import numpy as np
import math

# Number of classes
NUM_CLASSES = 3


def fisher_discriminant_features(x, y):

    num_data = len(x)
    dim_feat = len(x[0])

    x_classified = [[], [], []]

    for (i, j) in zip(x, y):
        x_classified[j-1].append(i)

    x_classified = [np.array(x_classified[0]).transpose(), np.array(x_classified[1]).transpose(), np.array(x_classified[2]).transpose()]

    x_mean = [x_classified[i].mean(axis=1) for i in range(3)]

    x_mean_total = 0
    for i in range(3):
        x_mean_total += x_mean[i] * x_classified[0].shape[1]

    x_mean_total /= num_data

    S_w = np.zeros((dim_feat, dim_feat))
    S_b = np.zeros((dim_feat, dim_feat))
    for i in range(3):
        S_w += (x_classified[i] - x_mean[i][:, np.newaxis]) @ (x_classified[i] - x_mean[i][:, np.newaxis]).transpose()
        S_b += x_classified[i].shape[1] * (x_mean[i][:, np.newaxis] - x_mean_total[:, np.newaxis]) @ (x_mean[i][:, np.newaxis] - x_mean_total[:, np.newaxis]).transpose()

    eig_val, eig_vec = np.linalg.eigh(np.linalg.inv(S_w) @ S_b)
    idx = np.absolute(eig_val).argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    Theta = (eig_vec[0] / np.linalg.norm(eig_vec[0]))[:, np.newaxis]
    Theta = np.append(Theta, (eig_vec[1] / np.linalg.norm(eig_vec[1]))[:, np.newaxis], axis=1)

    # print(Theta.shape)

    return Theta, x_mean


def classify(dist_func, x, df_param, class_centroid):

    min_dist = math.inf
    min_idx = 0

    for i in range(3):
        x_df = df_param.transpose() @ x
        cc_df = df_param.transpose() @ class_centroid[i]

        dist = dist_func(x_df, cc_df)

        if dist < min_dist:
            min_dist = dist
            min_idx = i+1

    return min_idx, min_dist


def test_stat(dist_func, test_data, df_param, class_centroids):

    x = test_data[1]
    y_true = test_data[0]
    num_test = len(x)

    accuracy = 0

    for i in range(num_test):
        c_idx, dist = classify(dist_func, np.array(x[i]), df_param, class_centroids)

        if c_idx == y_true[i]:
            accuracy += 1

    return accuracy / num_test


def accuracy_test(t_size, add_noise=False, sigma_sqr=None, ret_acc=True):

    acc = 0

    for i in range(NUM_RANDOM):

        train_data, test_data = load_train_test_data(t_size, True)

        if add_noise:
            train_data = add_gaussian_noises(train_data, sigma_sqr)

        df_param, c_centroids = fisher_discriminant_features(train_data[1], train_data[0])

        acc += test_stat(euclidean_dist, test_data, df_param, c_centroids)

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

    train_data, test_data = load_train_test_data()

    df_param, c_centroids = fisher_discriminant_features(train_data[1], train_data[0])

    test_stat(euclidean_dist, test_data, df_param, c_centroids)
