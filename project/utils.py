"""
Project utility functions

author: Yajue Yang
"""

import numpy as np

NUM_RANDOM = 1


def euclidean_dist(x, y):
    return np.linalg.norm((np.array(x) - np.array(y)))