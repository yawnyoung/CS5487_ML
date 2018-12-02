"""
Part1 solutions

author: Yajue Yang
"""

from PA_2.data_processing import *
from PA_2.clustering_alg import k_means


if __name__ == '__main__':
    x, y = load_synthetic_data('A')

    init_mean = np.array([[-3, 3, 4, -2], [3, 4, -3, -2]])
    epsilon = np.array([0.001, 0.001, 0.001, 0.001])

    curr_mean, z = k_means(x, init_mean, epsilon)

    print(curr_mean)