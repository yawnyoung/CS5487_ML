"""
Part2 solution

author Yajue Yang
"""

from PA_1.data_processing import load_count_data
from PA_1.regression_alg import *
from PA_1.result_analysis import *


def compare_algorithms(sample_x, sample_y, test_x, test_y):

    theta = 0
    mu_e = 0
    var_e = 0

    algo_name = ['Least Squares', 'Regularized Least Squares', 'LASSO', 'Robust Regression', 'Bayesian Regression']

    for i in range(5):
        if i == 0:
            theta = LS_regression(sample_x, np.array(sample_y))
        elif i == 1:
            theta = RLS_regression(sample_x, np.array(sample_y), 5)
        elif i == 2:
            theta = LASSO_regression(sample_x, np.array(sample_y), 5)
        elif i == 3:
            theta = RR_regression(sample_x, np.array(sample_y))
        else:
            mu_e, var_e = BR_regression(sample_x, sample_y, 10)

        if i == 4:
            et_y = np.transpose(test_x) @ mu_e
            var_star = np.transpose(test_x) @ var_e @ test_x
            # std_err = np.sqrt(var_star.diagonal())
        else:
            et_y = np.transpose(test_x) @ theta

        et_y = np.round(et_y)
        plot_predicted_vs_true(et_y, test_y, algo_name[i])

        print(algo_name[i] + ' mse: ', mean_square_err(et_y, test_y))
        print(algo_name[i] + ' mae: ', mean_abs_err(et_y, test_y))


def no_transform_test():

    train_x, train_y = load_count_data()
    test_x, test_y = load_count_data(sampled=False)

    compare_algorithms(train_x, train_y, test_x, test_y)


def sec_order_tf(x):

    modified_x = np.array(x, copy=True)
    x_square = np.square(x)

    modified_x = np.append(modified_x, x_square, axis=0)

    return modified_x


def sec_order_tf_test():
    train_x, train_y = load_count_data()
    test_x, test_y = load_count_data(sampled=False)

    train_x = sec_order_tf(train_x)
    test_x = sec_order_tf(test_x)

    compare_algorithms(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    no_transform_test()
    # sec_order_tf_test()