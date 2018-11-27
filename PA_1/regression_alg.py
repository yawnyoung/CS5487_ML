"""
Regression algorithms used in PA-1

author: Yajue Yang
"""

import numpy as np
import scipy.optimize
from cvxopt import matrix, solvers


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


def solve_qp_scipy(G, a, C, b):
    """
    Minimize 1/2 x^T G x - a^T x, s.t. C.T x >= b
    :return: optimization results
    """
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


def solve_lp_scipy(C, A_ub, b_ub, A_eq, b_eq, bounds=None):
    """
    Minimize C^T * x, s.t. A_ub * x <= b_ub and A_eq * x == b_eq
    :return: optimization results
    """
    result = scipy.optimize.linprog(C, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point')

    return result


def solve_lp_cvxopt(C, A_ub, b_ub, A_eq=None, b_eq=None):

    c = matrix(C)
    G = matrix(A_ub)
    h = matrix(b_ub)

    A = None
    if A_eq:
        A = matrix(A_eq)

    b = None
    if b_eq:
        b = matrix(b_eq)

    result = solvers.lp(c, G, h, A, b)

    x_sol = result['x']

    return x_sol


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


def RR_regression(x, y):
    """
    Robust regression method
    :param x: the batch of transformed inputs [x1, x2, ..., xn]
    :param y: the batch of observations [y1; y2; ...; yn]
    :return: fitted parameters
    """

    dim_param = x.shape[0]
    num_samples = x.shape[1]

    C = np.zeros(dim_param)

    C = np.concatenate((C, np.ones(num_samples)))

    A_ub = np.ndarray((2 * num_samples, dim_param + num_samples))

    x_t = np.transpose(x)

    A_ub[:num_samples, :dim_param] = -x_t
    A_ub[:num_samples, dim_param:] = -np.identity(num_samples)
    A_ub[num_samples:, :dim_param] = x_t
    A_ub[num_samples:, dim_param:] = -np.identity(num_samples)

    b_ub = -y
    b_ub = np.concatenate((b_ub, y))

    cvx_x_sol = solve_lp_cvxopt(C, A_ub, b_ub)

    theta = np.ndarray((dim_param,))
    for i in range(dim_param):
        theta[i] = cvx_x_sol[i]

    return theta


def BR_regression(x, y, hp):
    """
    Bayesian regression. theta ~ N(0, alpha * I), y|x, theta ~ N(f(x, theta), sigma^2)
    :param x: the batch of transformed inputs [x1, x2, ..., xn]
    :param y: the batch of observations [y1; y2; ...; yn]
    :param hp: hyper-parameter
    :return: fitted parameters
    """
    dim_param = x.shape[0]
    x_t = np.transpose(x)

    # estimated variance
    var_e = np.linalg.inv((1 / hp) * np.identity(dim_param) + (1 / 5) * x @ x_t)

    # estimated variance
    mu_e = (1 / 5) * var_e @ x @ y

    return mu_e, var_e
