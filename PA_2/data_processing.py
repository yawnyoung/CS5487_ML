"""
Data processing for the programming assignment 2

author: Yajue Yang
"""

import numpy as np


def load_synthetic_data(data_name):
    """
    Load synthetic data with the given name
    :param data_name: name of the data
    :return: (x, y)
    """

    assert (data_name == 'A' or data_name == 'B' or data_name == 'C')

    # load x
    file_name = 'cluster_data_data' + data_name + '_X.txt'

    x = []
    with open(file_name) as f:
        for line in f:
            line = line.split()
            x.append([float(v) for v in line])

    # load y
    file_name = 'cluster_data_data' + data_name + '_Y.txt'

    y = []
    with open(file_name) as f:
        for line in f:
            line = line.split()
            y.append([float(v) for v in line])

    x = np.array(x)
    y = np.array(y)

    return x, y


if __name__ == '__main__':
    load_synthetic_data('B')