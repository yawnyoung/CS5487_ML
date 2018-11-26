"""
Part1 solution

author Yajue Yang
"""

from PA_1.data_processing import load_polydata
from PA_1.data_processing import feature_poly_tf
from PA_1.regression_alg import *
import matplotlib.pyplot as plt


def estimated_func_plot(t_x, t_y, et_y, s_x, es_y):
    """
    Plot estimated function
    :param t_x: polyx
    :param t_y: true polyy
    :param et_y: estimated polyy
    :param s_x: samplex
    :param es_y: estimated sampley
    :return:
    """
    plt.plot(t_x, t_y, 'r', t_x, et_y, 'b.', s_x, es_y, 'g.')
    plt.show()


if __name__ == '__main__':

    sample_x, sample_y = load_polydata()
    test_x, test_y = load_polydata(False)

    # 5th order polynomial inputs
    sample_phi_5 = feature_poly_tf(sample_x, 5)
    test_phi_5 = feature_poly_tf(test_x, 5)

    """
    LS training
    """
    # theta_ls_5 = LS_regression(sample_phi_5, np.array(sample_y))
    #
    # et_y = np.transpose(test_phi_5) @ theta_ls_5
    #
    # es_y = np.transpose(sample_phi_5) @ theta_ls_5
    #
    # estimated_func_plot(test_x, test_y, et_y, sample_x, es_y)

    """
    RLS training
    """
    theta_rls_5 = RLS_regression(sample_phi_5, np.array(sample_y), 0.01)

    et_y = np.transpose(test_phi_5) @ theta_rls_5

    es_y = np.transpose(sample_phi_5) @ theta_rls_5

    estimated_func_plot(test_x, test_y, et_y, sample_x, es_y)