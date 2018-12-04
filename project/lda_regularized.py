"""
Regularized linear regression with LS minimization

author: Yajue
"""

import numpy as np
from project.data_processing import load_train_test_data, add_gaussian_noises
from project.lda_ls import get_data_matrix, decision_rule, test_stat
from project.utils import *


def train_w(train_data_mat, hp):

    y = train_data_mat[0]
    x = train_data_mat[1]

    dim_param = x.shape[1]

    w = np.linalg.inv(x.T @ x + hp * np.identity(dim_param)) @ x.T @ y

    return w


def classify_data(w, feat):

    y_classfied = w.T @ feat

    return decision_rule(y_classfied)


def accuracy_against_size(t_size, hp, add_noise=False, sigma_sqr=None, ret_acc=True):

    acc = 0

    for i in range(NUM_RANDOM):
        train_data, test_data = load_train_test_data(t_size, True)

        if add_noise:
            train_data = add_gaussian_noises(train_data, sigma_sqr)

        train_data_mat = get_data_matrix(train_data)

        w = train_w(train_data_mat, hp)

        acc += test_stat(w, test_data)

    acc /= NUM_RANDOM

    if ret_acc:
        return acc
    else:
        return 1 - acc


def accuracy_against_size_stat(train_sizes, hp, add_noise=False, sigma_sqr=None, ret_acc=True):

    acc = []

    for s in train_sizes:
        acc.append(accuracy_against_size(s, hp, add_noise, sigma_sqr, ret_acc))

    return acc


if __name__ == '__main__':

    train_data, test_data = load_train_test_data(0.2)

    train_data_mat = get_data_matrix(train_data)

    w = train_w(train_data_mat, 100)

    test_stat(w, test_data)
