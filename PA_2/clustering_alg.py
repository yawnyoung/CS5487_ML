"""
Clustering algorithms

author: Yajue
"""

import numpy as np
from scipy.stats import multivariate_normal
import math
from PA_2.data_processing import *


def euclidean_dist(x, y):
    return np.linalg.norm((np.array(x) - np.array(y)))


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

        # dists = np.array([np.linalg.norm(x[:2, i] - curr_mean[:2, k]) + 0.1 * np.linalg.norm(x[2:, i] - curr_mean[2:, k])
        #                   for k in range(num_class)])

        min_idx = np.argmin(dists)

        z[min_idx, i] = 1

    return z


def update_kmeans_mean(x, z, mean_range = None):

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

    # print('count class: ', count_class)

    for k in range(num_class):

        if count_class[k] != 0:
            updated_mean[:, k] /= count_class[k]

        else:
            for d in range(dim_feat):
                updated_mean[d, k] = np.random.uniform(mean_range[d, 0], mean_range[d, 1], 1)
    # print('mean: \n', updated_mean)

    return updated_mean


def k_means(x, init_mean, epsilon, mean_range = None):
    """
    K-Means clustering algorithm
    :param x: data
    :param init_mean: initial means
    :param epsilon: error threshold
    :return: means and labels of each data
    """

    num_class = init_mean.shape[1]
    num_data = x.shape[1]

    mean_err = np.full(num_class, np.inf)

    z = np.zeros((num_class, num_data))
    curr_mean = np.copy(init_mean)

    iteration = 0

    while one_true(np.greater(mean_err, epsilon)) and iteration < 1000:
        print('iteration: ', iteration)
        # print('current mean: ', curr_mean)

        z = update_kmeans_z(x, curr_mean)

        new_mean = update_kmeans_mean(x, z, mean_range)

        mean_err = np.array([np.linalg.norm(new_mean[:, k] - curr_mean[:, k]) for k in range(num_class)])

        curr_mean = new_mean

        iteration += 1

    return curr_mean, z


def update_gmm_z(x, curr_pis, curr_means, curr_cov):

    num_class = curr_pis.size
    num_data = x.shape[1]

    z = np.zeros((num_class, num_data))

    for i in range(num_data):

        for k in range(num_class):

            z_numerator = curr_pis[k] * multivariate_normal.pdf(x[:, i],
                                                                mean=curr_means[:, k],
                                                                cov=curr_cov[:, :, k])

            z_den = 0

            for j in range(num_class):
                z_den += curr_pis[j] * multivariate_normal.pdf(x[:, i], mean=curr_means[:, j], cov=curr_cov[:, :, j])

            # print('num: {}, den: {}'.format(z_numerator, z_den))

            z[k, i] = z_numerator / z_den

    return z


def update_gmm_params(x, z):

    num_class = z.shape[0]
    num_data = x.shape[1]
    dim_feat = x.shape[0]

    count_class = np.array([np.sum(z[k, :]) for k in range(num_class)])

    new_pis = count_class / num_data

    new_means = np.zeros((dim_feat, num_class))
    new_covs = np.zeros((dim_feat, dim_feat, num_class))
    for k in range(num_class):

        for i in range(num_data):
            new_means[:, k] += z[k, i] * x[:, i]

        new_means[:, k] /= count_class[k]

        for i in range(num_data):
            mean_diff = (x[:, i] - new_means[:, k])[:, np.newaxis]
            new_covs[:, :, k] += np.diag(np.diag(z[k, i] * (mean_diff @ np.transpose(mean_diff))))

        # print(new_means[:, k])
        new_covs[:, :, k] /= count_class[k]

    return new_pis, new_means, new_covs


def gmm_log_likelihood(x, curr_pis, curr_means, curr_covs):

    num_class = curr_means.shape[1]
    num_data = x.shape[1]

    log_likelihood = 0

    for i in range(num_data):

        likelihood = 0

        for k in range(num_class):

            likelihood += curr_pis[k] * multivariate_normal.pdf(x[:, i], curr_means[:, k], curr_covs[:, :, k])

        log_likelihood += np.log(likelihood)

    return log_likelihood


def EM_GMM(x, init_pis, init_means, init_cov, inc_thld):

    curr_pis = np.copy(init_pis)
    curr_means = np.copy(init_means)
    curr_cov = np.copy(init_cov)

    q = gmm_log_likelihood(x, curr_pis, curr_means, curr_cov)

    q_inc = math.inf

    num_class = init_pis.size
    num_data = x.shape[1]

    z = np.zeros((num_class, num_data))

    while q_inc > inc_thld:

        z = update_gmm_z(x, curr_pis, curr_means, curr_cov)
        curr_pis, curr_means, curr_cov = update_gmm_params(x, z)

        q_new = gmm_log_likelihood(x, curr_pis, curr_means, curr_cov)

        q_inc = q_new - q

        q = q_new

        print('q: {}, q_inc: {}'.format(q, q_inc))

    print(curr_pis)
    print(curr_means)

    return curr_pis, curr_means, curr_cov, z


def neighbourhood_points(x, data, d_thold=2):

    num_data = data.shape[1]

    x_nbr = []

    for i in range(num_data):
        distance_between = euclidean_dist(x, data[:, i])
        if distance_between <= d_thold:
            x_nbr.append(data[:, i].tolist())

    x_nbr = np.array(x_nbr).T

    # print(x_nbr.shape)

    return x_nbr


def gaussian_kernel(dist, h):

    val = (1 / (h * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((dist / h)) ** 2)

    return val


def mean_shift(x, h, x_init, ct):

    num_data = x.shape[1]
    dim_param = x.shape[0]

    x_err = math.inf

    x_curr = np.copy(x_init)
    print('before updating: ', x_curr)

    while x_err > ct:

        # find neighbor points
        x_nbr = neighbourhood_points(x_curr, x, h)

        num_nbr = x_nbr.shape[1]
        print('# neighbors: ', num_nbr)

        x_expectation = np.zeros(dim_param)

        den = 0
        for i in range(num_nbr):
            cov = h * h * np.identity(dim_param)
            cov[2, 2] = 10 * h * h
            cov[3, 3] = 10 * h * h
            x_expectation += x_nbr[:, i] * multivariate_normal.pdf(x_nbr[:, i], x_curr, cov)
            den += multivariate_normal.pdf(x_nbr[:, i], x_curr, cov)
            # x_expectation += x_nbr[:, i] * multivariate_normal.pdf(x_nbr[:, i], x_curr, h * h * np.identity(dim_param))
            # den += multivariate_normal.pdf(x_nbr[:, i], x_curr, h * h * np.identity(dim_param))

        x_next = x_expectation / den

        x_err = np.linalg.norm(x_curr - x_next)

        x_curr = x_next

    print('after updating: ', x_curr)

    return x_curr


def mean_shift_clustering(x, h, ct, pr_min):

    num_data = x.shape[1]

    x_cnvg = np.ndarray(x.shape)

    peaks = []
    z = np.ndarray(num_data)

    for i in range(num_data):
        print(i)

        x_cnvg[:, i] = mean_shift(x, h, x[:, i], ct)

        if len(peaks) == 0:
            peaks.append(x_cnvg[:, i])
            z[i] = 0

        else:
            dists = np.array([np.linalg.norm(p_v - x_cnvg[:, i]) for p_v in peaks])
            if np.min(dists) < pr_min:
                z[i] = np.argmin(dists)
            else:
                peaks.append(x_cnvg[:, i])
                z[i] = len(peaks) - 1

    # print(x_cnvg)

    return x_cnvg, peaks, z


if __name__ == '__main__':

    data_name = 'A'

    x, y = load_synthetic_data(data_name)

    mean_shift(x, 0, x[:, 0], 0)