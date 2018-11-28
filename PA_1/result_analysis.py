"""
Result analysis

author: Yaju Yang
"""

import matplotlib.pyplot as plt
import numpy as np


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


def plot_predicted_vs_true(et_y, t_y, title=None):
    """
    Plot predictions and true counts
    :param et_y:
    :param t_y:
    :return:
    """
    plt.plot(et_y, 'r', label='predictions')
    plt.plot(t_y, 'b', label='true counts')
    if title:
        plt.title(title)
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


def mean_abs_err(y_t, y_e):
    """
    Calculate mean absolute error
    :param y_t: true y
    :param y_e: estimated y
    :return:
    """
    return np.mean(np.absolute(y_e - y_t))