"""
Compare results of various algorithms

author: Yajue Yang
"""

import numpy as np
import project.qda as qda
import project.lda_ls as lda
import project.lda_regularized as rlda
import project.lda_fisher as flda
import project.knn as knn
import matplotlib.pyplot as plt
import matplotlib.cm as cm


ALG_NAME = ['QDA', 'LDA', 'RLDA', 'FLDA', 'KNN', 'QSVM', 'CSVM', 'LSVM', 'LR']

# COLORS = cm.rainbow(np.linspace(0, 1, len(ALG_NAME)))
COLORS = cm.tab10(np.linspace(0, 1, len(ALG_NAME)))

acc_size_svm_q = [0.9209876543, 0.9270833333, 0.9246031746, 0.9388888889, 0.9533333333, 0.9493150685, 0.9563636364, 0.9405405405, 0.9684210526]
acc_size_svm_c = [0.9172839506, 0.9256944444, 0.9182539683, 0.937962963, 0.9522222222, 0.9452054795, 0.96, 0.9513513514, 0.9736842105]
acc_size_svm_l = [0.924691358, 0.9284722222, 0.923015873, 0.9194444444, 0.9488888889, 0.9547945205, 0.9618181818, 0.9432432432, 0.9526315789]
acc_size_lr = [0.925308642, 0.9291666667, 0.9325396825, 0.95, 0.9622222222, 0.9643835616, 0.9563636364, 0.9756756757, 0.9631578947]


acc_noise_lr = [0.8611111111, 0.87, 0.87, 0.7788888889]
acc_noise_svm_q = [0.9388888889, 0.94, 0.9022222222, 0.9233333333]
acc_noise_svm_c = [0.9422222222, 0.9466666667, 0.9144444444, 0.9211111111]
acc_noise_svm_l = [0.9322222222, 0.9166666667, 0.9166666667, 0.9122222222]


def comp_acc_vs_sizes():

    train_sizes = np.arange(0.1, 1.0, 0.1)

    plot_acc = True

    accuracy = []

    accuracy.append(qda.accuracy_against_size_stat(train_sizes, ret_acc=plot_acc))
    accuracy.append(lda.accuracy_against_size_stat(train_sizes, ret_acc=plot_acc))
    accuracy.append(rlda.accuracy_against_size_stat(train_sizes,  5, ret_acc=plot_acc))
    accuracy.append(flda.accuracy_against_size_stat(train_sizes, ret_acc=plot_acc))
    accuracy.append(knn.accuracy_against_size_stat(train_sizes, ret_acc=plot_acc))
    accuracy.append(acc_size_svm_q)
    accuracy.append(acc_size_svm_c)
    accuracy.append(acc_size_svm_l)
    accuracy.append(acc_size_lr)

    for i in range(len(accuracy)):
        plt.plot(train_sizes, accuracy[i], color=COLORS[i], label=ALG_NAME[i])

    plt.xlabel('Size of Training Data')
    if plot_acc:
        plt.ylabel('Accuracy')
    else:
        plt.ylabel('Error')
    plt.legend()
    plt.show()


def comp_acc_vs_noises():

    sigma_sqr = np.arange(1, 20, 5)
    train_size = 0.5
    # print(sigma_sqr)

    plot_acc = True

    accuracy = []

    accuracy.append(qda.accuracy_against_noise_stat(sigma_sqr, train_size=train_size, ret_acc=plot_acc))
    accuracy.append(lda.accuracy_against_noise_stat(sigma_sqr, train_size=train_size, ret_acc=plot_acc))
    accuracy.append(rlda.accuracy_against_noise_stat(sigma_sqr, hp=0.1, train_size=train_size, ret_acc=plot_acc))
    accuracy.append(flda.accuracy_against_noise_stat(sigma_sqr, train_size=train_size, ret_acc=plot_acc))
    accuracy.append(knn.accuracy_against_noise_stat(sigma_sqr, train_size=train_size, ret_acc=plot_acc))
    accuracy.append(acc_noise_svm_q)
    accuracy.append(acc_noise_svm_c)
    accuracy.append(acc_noise_svm_l)
    accuracy.append(acc_noise_lr)

    for i in range(len(accuracy)):
        plt.plot(sigma_sqr, accuracy[i], color=COLORS[i], label=ALG_NAME[i])

    plt.xlabel('Noise Mean')
    if plot_acc:
        plt.ylabel('Accuracy')
    else:
        plt.ylabel('Error')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # comp_acc_vs_sizes()

    comp_acc_vs_noises()