"""
Regression algorithms used in PA-1

author: Yajue Yang
"""

import numpy as np
import scipy.optimize


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
    Regularized Least Square regression method
    :param x: the batch of transformed inputs [x1, x2, ..., xn]
    :param y: the batch of observations [y1; y2; ...; yn]
    :param hp: hyperparameter
    :return: fitted parameters
    """
    x_t = np.transpose(x)

    dim_param = x.shape[0]

    I_mat = np.identity(dim_param)

    theta = np.linalg.inv(x @ x_t + I_mat * hp) @ x @ y

    return theta


def solve_qp_scipy(G, a, C, b, meq=0):
    # Minimize     1/2 x^T G x - a^T x
    # Subject to   C.T x >= b
    def f(x):
        return 0.5 * np.dot(x, G).dot(x) - np.dot(a, x)

    if C is not None and b is not None:
        constraints = [{
            'type': 'ineq',
            'fun': lambda x, C=C, b=b, i=i: (np.dot(C.T, x) - b)[i]
        } for i in range(C.shape[1])]
    else:
        constraints = []

    result = scipy.optimize.minimize(f, x0=np.zeros(len(G)), method='COBYLA', constraints=constraints, tol=1e-10)
    return result


def LASSO_regression(x, y, hp):
    """
    LASSO regression method (min 1/2 theta^T * H * theta - f^T * theta, C^T * theta >= 0)
    :param x: the batch of transformed inputs [x1, x2, ..., xn]
    :param y: the batch of observations [y1; y2; ...; yn]
    :param hp: hyperparameter
    :return: fitted parameters
    """
    dim_param = x.shape[0]

    x_t = np.transpose(x)

    H = np.ndarray((2 * dim_param, 2 * dim_param))

    H[:dim_param, :dim_param] = x @ x_t
    H[:dim_param, dim_param:] = - x @ x_t
    H[dim_param:, :dim_param] = - x @ x_t
    H[dim_param:, dim_param:] = x @ x_t

    f = x @ y
    f = np.concatenate((f, - x @ y))
    f = f - hp * np.ones(2 * dim_param)

    C = np.identity(2 * dim_param)

    b = np.zeros(2 * dim_param)

    result = solve_qp_scipy(H, f, C, b)

    theta_pm = result.x
    theta = [theta_pm[i] - theta_pm[i+dim_param] for i in range(dim_param)]

    return theta


# TODO Robust regression