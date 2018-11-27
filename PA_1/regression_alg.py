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
    result = scipy.optimize.linprog(C, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method='interior-point',
                                    options={'alpha0': 0.99999, 'beta': 0.1, 'maxiter': 100000, 'disp': False,
                                             'tol': 1e-10, 'sparse': False, 'lstsq': False, 'sym_pos': True,
                                             'cholesky': None, 'pc': True, 'ip': False, 'presolve': False,
                                             'permc_spec': 'MMD_AT_PLUS_A', 'rr': True, '_sparse_presolve': False})

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


def RR_regression(x, y):
    """
    Robust regression method
    :param x: the batch of transformed inputs [x1, x2, ..., xn]
    :param y: the batch of observations [y1; y2; ...; yn]
    :return: fitted parameters
    """

    dim_param = x.shape[0]
    num_samples = x.shape[1]

    print(x.shape)

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

    # print(b_ub)

    result = solve_lp_scipy(C, A_ub, b_ub, A_eq=None, b_eq=None)

    aux_theta = result.x

    print(aux_theta)

    theta = aux_theta[:dim_param]

    print(theta)

    return theta