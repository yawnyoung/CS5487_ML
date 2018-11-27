"""
Part1 solution

author Yajue Yang
"""

from PA_1.data_processing import load_polydata
from PA_1.data_processing import feature_poly_tf
from PA_1.regression_alg import *
import matplotlib.pyplot as plt


def estimated_func_plot(t_x, et_y, s_x, s_y, title=None):
    """
    Plot estimated function
    :param t_x: polyx
    :param et_y: estimated polyy
    :param s_x: samplex
    :param s_y: sampley
    :return:
    """
    plt.plot(t_x, et_y, 'r', label='estimated function')
    plt.plot(s_x, s_y, 'b.', label='sample data')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def br_func_plot(t_x, et_y, std_err, s_x, s_y):
    """
    Plot Bayesian estimated function
    :param t_x: polyx
    :param et_y: estimated polyy
    :param std_err: standard deviation
    :param s_x: samplex
    :param s_y: sampley
    :return:
    """
    plt.plot(t_x, et_y, 'r', label='estimated function')
    plt.fill_between(t_x, et_y-std_err, et_y+std_err, label='standard deviation', color='g')
    plt.plot(s_x, s_y, 'b.', label='sample data')
    plt.title('Bayesian Regression')
    plt.legend()
    plt.show()


def mean_square_err(y_t, y_e):
    """
    Calculate mean square error
    :param y_t: true y
    :param y_e: estimated y
    :return:
    """
    return np.mean(np.square(y_e - y_t))


if __name__ == '__main__':

    ratio = 0.15

    sample_x, sample_y = load_polydata(ratio)
    test_x, test_y = load_polydata(1, False)

    # 5th order polynomial inputs
    sample_phi = feature_poly_tf(sample_x, 5)
    test_phi = feature_poly_tf(test_x, 5)

    """
    LS
    """
    theta_ls = LS_regression(sample_phi, np.array(sample_y))

    et_y = np.transpose(test_phi) @ theta_ls

    print('ls mse: ', mean_square_err(et_y, test_y))

    # estimated_func_plot(test_x, et_y, sample_x, sample_y, 'Least Squares')

    """
    RLS
    """
    theta_rls = RLS_regression(sample_phi, np.array(sample_y), 5)

    et_y = np.transpose(test_phi) @ theta_rls

    print('rls mse: ', mean_square_err(et_y, test_y))

    # estimated_func_plot(test_x, et_y, sample_x, sample_y, 'Regularized Least Squares')

    """
    LASSO
    # """
    theta_lasso = LASSO_regression(sample_phi, np.array(sample_y), 5)

    et_y = np.transpose(test_phi) @ theta_lasso

    print('lasso mse: ', mean_square_err(et_y, test_y))

    # estimated_func_plot(test_x, et_y, sample_x, sample_y, 'LASSO')

    """
    RR
    """
    theta_rr = RR_regression(sample_phi, np.array(sample_y))

    et_y = np.transpose(test_phi) @ theta_rr

    # estimated_func_plot(test_x, et_y, sample_x, sample_y, 'Robust Regression')

    # print('rr mse: ', mean_square_err(et_y, test_y))

    """
    BR
    """
    mu_e, var_e = BR_regression(sample_phi, sample_y, 10)
    et_y = np.transpose(test_phi) @ mu_e
    print('br mse: ', mean_square_err(et_y, test_y))
    # var_star = np.transpose(test_phi) @ var_e @ test_phi
    # std_err = np.sqrt(var_star.diagonal())
    # br_func_plot(test_x, et_y, std_err, sample_x, sample_y)