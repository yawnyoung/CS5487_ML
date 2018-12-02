"""
Clustering algorithms

author: Yajue
"""

import numpy as np


def one_true(conds):
    """
    If one of the conditions is true, return true;
    Else return false.
    :param conds: conditions
    :return:
    """

    for c in conds:
        if c:
            return True

    return False


def update_kmeans_z(x, curr_mean):

    num_class = curr_mean.shape[1]
    num_data = x.shape[1]

    z = np.zeros((num_class, num_data))

    for i in range(num_data):

        dists = np.array([np.linalg.norm(x[:, i] - curr_mean[:, k]) for k in range(num_class)])

        min_idx = np.argmin(dists)

        z[min_idx, i] = 1

    return z


def update_kmeans_mean(x, z):

    num_class = z.shape[0]
    num_data = x.shape[1]
    dim_feat = x.shape[0]

    updated_mean = np.zeros((dim_feat, num_class))
    count_class = np.zeros(num_class)

    for i in range(num_data):

        for k in range(num_class):
            if z[k, i] == 1:
                updated_mean[:, k] += x[:, i]
                count_class[k] += 1
                break

    updated_mean = np.divide(updated_mean, count_class)

    return updated_mean


def k_means(x, init_mean, epsilon):

    num_class = init_mean.shape[1]
    num_data = x.shape[1]

    mean_err = np.full(num_class, np.inf)

    z = np.zeros((num_class, num_data))
    curr_mean = np.copy(init_mean)

    while one_true(np.greater(mean_err, epsilon)):

        z = update_kmeans_z(x, curr_mean)

        new_mean = update_kmeans_mean(x, z)

        mean_err = np.array([np.linalg.norm(new_mean[:, k] - curr_mean[:, k]) for k in range(num_class)])

        curr_mean = new_mean

    return curr_mean, z





