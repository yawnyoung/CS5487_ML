"""
Project utility functions

author: Yajue Yang
"""

import numpy as np

NUM_RANDOM = 5
SIGMA_SQR = 0.1

def euclidean_dist(x, y):
    return np.linalg.norm((np.array(x) - np.array(y)))