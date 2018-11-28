"""
Part1 solution

author Yajue Yang
"""

from PA_1.data_processing import load_polydata
from PA_1.data_processing import feature_poly_tf
from PA_1.regression_alg import *
import random
from PA_1.result_analysis import *


def err_vs_training_size():
    """
    Find the error versus training size
    :return:
    """
    ratio = [1, 0.85, 0.75, 0.6, 0.5, 0.25, 0.15]

    mse_stat = np.ndarray((5, len(ratio)))

    num_trials = 10

    for alg_idx in range(5):

        for r_idx in range(len(ratio)):

            mse = 0
            theta = 0
            mu_e = 0

            for trial_idx in range(num_trials):
                sample_x, sample_y = load_polydata(ratio[r_idx], shuffle=True)
                test_x, test_y = load_polydata(1, shuffle=False)
                sample_phi = feature_poly_tf(sample_x, 5)
                test_phi = feature_poly_tf(test_x, 5)
                if alg_idx == 0:
                    theta = LS_regression(sample_phi, np.array(sample_y))
                elif alg_idx == 1:
                    theta = RLS_regression(sample_phi, np.array(sample_y), 5)
                elif alg_idx == 2:
                    theta = LASSO_regression(sample_phi, np.array(sample_y), 5)
                elif alg_idx == 3:
                    theta = RR_regression(sample_phi, np.array(sample_y))
                else:
                    mu_e, var_e = BR_regression(sample_phi, sample_y, 10)

                if alg_idx == 4:
                    et_y = np.transpose(test_phi) @ mu_e
                else:
                    et_y = np.transpose(test_phi) @ theta

                mse += mean_square_err(et_y, test_y)

            mse_stat[alg_idx, r_idx] = mse / num_trials

        print(alg_idx, r_idx, mse_stat[alg_idx, r_idx])

    # plot
    fig, axes = plt.subplots(5)

    colors = ['r', 'g', 'b', 'y', 'm']
    labels = ['LS', 'RLS', 'LASSO', 'RR', 'BR']
    for i in range(5):
        axes[i].plot(ratio, mse_stat[i, :], colors[i], label=labels[i])
        axes[i].legend()

    fig.text(0.5, 0.04, "training size", ha="center", va="center")
    fig.text(0.05, 0.5, "mean-squared error", ha="center", va="center", rotation=90)

    plt.show()

    for i in range(5):
        plt.plot(ratio, mse_stat[i, :], colors[i], label=labels[i])

    plt.xlabel('training size')
    plt.ylabel('mean-squared error')

    plt.legend()
    plt.show()


def add_outliers(s_y, ratio=0.1, mu=5, std=5):
    """
    Add outliers output values with Gaussian noise
    :param s_y: sampley
    :param ratio: the size of values to be modified
    :return: modified y
    """
    num_sample = len(s_y)
    num_modified = int(num_sample * ratio)

    print('number of modified data: ', num_modified)

    indices_modified = random.sample(range(num_sample), num_modified)

    print('modified indices: ', indices_modified)

    modified_y = s_y.copy()

    for i in indices_modified:
        noise = np.random.normal(mu, std, 1)
        # print('noise: ', noise)
        modified_y[i] += noise
        # print(s_y[i], modified_y[i])

    return modified_y


def compare_algorithms(sample_x, sample_y, test_x, test_y, order):

    # 5th order polynomial inputs
    sample_phi = feature_poly_tf(sample_x, order)
    test_phi = feature_poly_tf(test_x, order)

    theta = 0
    mu_e = 0
    var_e = 0

    algo_name = ['Least Squares', 'Regularized Least Squares', 'LASSO', 'Robust Regression', 'Bayesian Regression']

    for i in range(5):
        if i == 0:
            theta = LS_regression(sample_phi, np.array(sample_y))
        elif i == 1:
            theta = RLS_regression(sample_phi, np.array(sample_y), 5)
        elif i == 2:
            theta = LASSO_regression(sample_phi, np.array(sample_y), 5)
        elif i == 3:
            theta = RR_regression(sample_phi, np.array(sample_y))
        else:
            mu_e, var_e = BR_regression(sample_phi, sample_y, 10)

        if i == 4:
            et_y = np.transpose(test_phi) @ mu_e
            var_star = np.transpose(test_phi) @ var_e @ test_phi
            std_err = np.sqrt(var_star.diagonal())
            br_func_plot(test_x, et_y, std_err, sample_x, sample_y)
        else:
            et_y = np.transpose(test_phi) @ theta
            estimated_func_plot(test_x, et_y, sample_x, sample_y, algo_name[i])

        print(algo_name[i] + ' mse: ', mean_square_err(et_y, test_y))


def outlier_test():

    sample_x, sample_y = load_polydata()
    test_x, test_y = load_polydata(sampled=False)
    sample_y = add_outliers(sample_y, ratio=0.1, mu=10, std=10)

    compare_algorithms(sample_x, sample_y, test_x, test_y, 5)


def ten_order_test():
    sample_x, sample_y = load_polydata()
    test_x, test_y = load_polydata(sampled=False)

    compare_algorithms(sample_x, sample_y, test_x, test_y, 10)


if __name__ == '__main__':

    """
    part1-c
    """
    # err_vs_training_size()

    """
    part1-d
    """
    # outlier_test()

    """
    part1-e
    """
    ten_order_test()