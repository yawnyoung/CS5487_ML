"""
Data processing for PA-1

author: Yajue Yang
"""

import numpy as np


def load_polydata(ratio = 1, sampled=True, shuffle=False):
    """
    load poly data
    :param sampled: if true, load sample data; if false, load test data
    :return: (x, y)
    """
    if sampled:
        file_name = 'polydata_data_samp'
    else:
        file_name = 'polydata_data_poly'

    # load x
    x = []
    with open(file_name + 'x.txt') as f:
        for line in f:
            line = line.split()
            x += [float(v) for v in line]

    # load y
    y = []
    with open(file_name + 'y.txt') as f:
        for line in f:
            y.append(float(line))

    x_y = np.ndarray((len(x), 2))

    x_y[:, 0] = np.array(x)
    x_y[:, 1] = np.array(y)

    if shuffle:
        np.random.shuffle(x_y)

    num_sample = int(ratio * len(x))

    x = x_y[:num_sample, 0]
    y = x_y[:num_sample, 1]

    return x, y


def feature_poly_tf(x, k):
    """
    Feature polynomial transformation
    :param x: feature
    :param k: the highest order of polynomial
    :return: transformed feature
    """
    n = len(x)
    Phi = np.ndarray(shape=(k+1, n), dtype=np.float32)

    for i in range(n):
        for j in range(k+1):
            Phi[j][i] = x[i]**j

    return Phi


if __name__ == '__main__':

    x, y = load_polydata()
    feature_poly_tf(x, 3)