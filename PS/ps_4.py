"""
Problem set 4

author: Yajue Yang
"""

import math
import numpy as np

D_L = np.array([229, 211, 93, 35, 7, 1])
D_A = np.array([325, 115, 67, 30, 18, 21])


def Poisson_dist(p_lambda, x):

    p = 1 / math.factorial(x) * math.exp(-p_lambda) * math.pow(p_lambda, x)

    return p


def get_num_hit(cell_idx, data):

    for i in range(1, len(data) + 1):
        if cell_idx < np.sum(data[:i]):
            return i-1


def update_z(x, j, p_lamdas, p_pis):
    K = p_pis.size

    z = p_pis[j] * Poisson_dist(p_lamdas[j], x)

    z_d = 0
    for k in range(K):
        z_d += p_pis[k] * Poisson_dist(p_lamdas[k], x)

    z /= z_d

    return z


def calc_expectation(data, Z, p_lambdas, p_pis):
    K = p_pis.size
    N = int(np.sum(data))

    q = 0
    for i in range(N):
        for j in range(K):
            x = get_num_hit(i, data)
            q += Z[i, j] * math.log1p(p_pis[j]) + Z[i, j] * math.log1p(Poisson_dist(p_lambdas[j], x))

    return q


def update_parameter(Z, data):

    N = Z.shape[0]
    K = Z.shape[1]

    p_pis = np.ndarray(K)
    p_lambdas = np.ndarray(K)

    for j in range(K):
        p_pis[j] = np.sum(Z[:, j]) / N

        p_lambdas[j] = 0

        for i in range(N):
            x = get_num_hit(i, data)
            p_lambdas[j] += Z[i, j] * x

        p_lambdas[j] /= np.sum((Z[:, j]))

    # print('lambda: ', p_lambdas)
    # print('pi: ', p_pis)

    return p_lambdas, p_pis


def update_Z(data, p_lamdas, p_pis):

    K = p_lamdas.size
    N = int(np.sum(data))

    Z = np.ndarray((N, K))

    for i in range(N):
        for j in range(K):
            x = get_num_hit(i, data)

            Z[i, j] = update_z(x, j, p_lamdas, p_pis)

    return Z


def EM_Poisson(init_lambda, init_pi, data):

    assert init_lambda.size == init_pi.size

    K = init_lambda.size
    N = int(np.sum(data))

    """
    Initializing E-step
    """
    Z = update_Z(data, init_lambda, init_pi)

    Q = calc_expectation(data, Z, init_lambda, init_pi)

    error = 100

    p_lamdas = init_lambda
    p_pis = init_pi

    while error > 0.01:

        """
        E step: update Z
        """
        Z = update_Z(data, p_lamdas, p_pis)

        """
        maximize Q
        """
        p_lamdas, p_pis = update_parameter(Z, data)

        Z = update_Z(data, p_lamdas, p_pis)

        Q_old = Q
        print('Q_old: ', Q_old)
        Q = calc_expectation(data, Z, p_lamdas, p_pis)
        print('Q: ', Q)

        error = Q - Q_old
        print('error:', error)

    print('lambdas: ', p_lamdas)
    print('pis: ', p_pis)


if __name__ == '__main__':

    # init_lambda = np.array([1])
    # init_pi = np.array([1])

    # init_lambda = np.array([1, 2, 3, 4, 5])
    # init_pi = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    #
    # print('London test')
    # EM_Poisson(init_lambda, init_pi, D_L)

    init_lambda = np.array([1, 2, 3, 4, 10])
    init_pi = np.array([0.6, 0.1, 0.1, 0.1, 0.1])

    print('Antwerp test')
    EM_Poisson(init_lambda, init_pi, D_A)
