"""
Regression algorithms used in PA-1

author: Yajue Yang
"""

import numpy as np


def LS_regression(x, y):
    """
    Least Squares regression method
    :param x: the batch of transformed inputs [x1, x2, ..., xn]
    :param y: the batch of observations [y1; y2; ...; yn]
    :return: fitted parameters
    """
    x_t = np.transpose(x)
    theta = np.linalg.inv(x @ x_t) @ x @ y
    return theta


def RLS_regression(x, y, hp):
    """
    Regularized Least Square regression model
    :param x: the batch of transformed inputs [x1, x2, ..., xn]
    :param y: the batch of observations [y1; y2; ...; yn]
    :param hp: hyperparameter
    :return:fitted parameters
    """
    x_t = np.transpose(x)

    dim_param = x.shape[0]

    I_mat = np.identity(dim_param)

    theta = np.linalg.inv(x @ x_t + I_mat * hp) @ x @ y

    return theta