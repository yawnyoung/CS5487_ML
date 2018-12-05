"""
Part2 Solutions

author: Yajue Yang
"""

import PA_2.pa2 as pa2_utils
from PA_2.clustering_alg import k_means, EM_GMM, mean_shift_clustering
import PIL.Image as Image
import numpy as np
import pylab as pl
from scipy.cluster.vq import whiten

IMAGE_NAME = ['images/12003.jpg', 'images/21077.jpg', 'images/56028.jpg', 'images/62096.jpg']


def test_k_means():
    img = Image.open(IMAGE_NAME[2])
    pl.subplot(1,3,1)
    pl.imshow(img)

    # extract features from image (step size = 7)
    X, L = pa2_utils.getfeatures(img, 7)

    dim_param = X.shape[0]
    num_class = 7
    init_mean = np.ndarray((dim_param, num_class))

    X_w = whiten(X.T).T
    X_minmax = np.ndarray((dim_param, 2))

    for i in range(dim_param):
        X_minmax[i, 0] = np.min(X_w[i, :])
        X_minmax[i, 1] = np.max(X_w[i, :])

    print(X_minmax)

    for i in range(dim_param):
        init_mean[i, :] = np.random.uniform(X_minmax[i, 0], X_minmax[i, 1], num_class).reshape((1, num_class))

    print(init_mean)

    epsilon = np.empty(num_class)
    epsilon.fill(0.001)

    means, Z = k_means(X_w, init_mean, epsilon, mean_range=X_minmax)

    num_data = X.shape[1]
    Y = np.array([np.argmax(Z[:, i]) + 1 for i in range(num_data)])
    print(Y)

    # make segmentation image from labels
    segm = pa2_utils.labels2seg(Y, L)
    pl.subplot(1,3,2)
    pl.imshow(segm)

    # color the segmentation image
    csegm = pa2_utils.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)
    pl.show()


def test_em_gmm():
    img = Image.open(IMAGE_NAME[0])
    pl.subplot(1, 3, 1)
    pl.imshow(img)

    # extract features from image (step size = 7)
    X, L = pa2_utils.getfeatures(img, 7)

    dim_param = X.shape[0]
    num_class = 7
    init_mean = np.ndarray((dim_param, num_class))

    X_w = whiten(X.T).T
    X_minmax = np.ndarray((dim_param, 2))

    init_pis = np.ndarray((num_class, 1))
    init_pis.fill(float(1) / float(num_class))

    for i in range(dim_param):
        X_minmax[i, 0] = np.min(X_w[i, :])
        X_minmax[i, 1] = np.max(X_w[i, :])

    print(X_minmax)

    for i in range(dim_param):
        init_mean[i, :] = np.random.uniform(X_minmax[i, 0], X_minmax[i, 1], num_class).reshape((1, num_class))

    print(init_mean)

    init_cov = np.ndarray((dim_param, dim_param, num_class))
    for i in range(num_class):
        init_cov[:, :, i] = np.identity(dim_param) * 0.1

    epsilon = 0.001

    pis, means, cov, Z = EM_GMM(X_w, init_pis, init_mean, init_cov, epsilon)

    num_data = X.shape[1]
    Y = np.array([np.argmax(Z[:, i]) + 1 for i in range(num_data)])
    print(Y)

    # make segmentation image from labels
    segm = pa2_utils.labels2seg(Y, L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)

    # color the segmentation image
    csegm = pa2_utils.colorsegms(segm, img)
    pl.subplot(1, 3, 3)
    pl.imshow(csegm)
    pl.show()


def test_mean_shift():
    img = Image.open(IMAGE_NAME[0])
    pl.subplot(1, 3, 1)
    pl.imshow(img)

    # extract features from image (step size = 7)
    X, L = pa2_utils.getfeatures(img, 7)

    X_w = whiten(X.T).T

    h = 1
    ct = 0.001
    pr_min = 0.1
    x_cnvg, peaks, Z = mean_shift_clustering(X_w, h, ct, pr_min)

    Y = Z + 1

    # make segmentation image from labels
    segm = pa2_utils.labels2seg(Y, L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)

    # color the segmentation image
    csegm = pa2_utils.colorsegms(segm, img)
    pl.subplot(1, 3, 3)
    pl.imshow(csegm)
    pl.show()


if __name__ == '__main__':

    # test_k_means()

    # test_em_gmm()

    test_mean_shift()