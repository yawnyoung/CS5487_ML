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


ALG_NAME = ['QDA', 'LDA', 'RLDA', 'FLDA', 'KNN']


def comp_acc_vs_sizes():

    train_sizes = np.arange(0.1, 1.0, 0.1)

    plot_acc = False

    accuracy = []

    accuracy.append(qda.accuracy_against_size_stat(train_sizes, ret_acc=plot_acc))
    accuracy.append(lda.accuracy_against_size_stat(train_sizes, ret_acc=plot_acc))
    accuracy.append(rlda.accuracy_against_size_stat(train_sizes,  5, ret_acc=plot_acc))
    accuracy.append(flda.accuracy_against_size_stat(train_sizes, ret_acc=plot_acc))
    accuracy.append(knn.accuracy_against_size_stat(train_sizes, ret_acc=plot_acc))

    colors = cm.rainbow(np.linspace(0, 1, len(accuracy)))

    for i in range(len(accuracy)):
        plt.plot(train_sizes, accuracy[i], color=colors[i], label=ALG_NAME[i])

    plt.xlabel('Size of Training Data')
    if plot_acc:
        plt.ylabel('Accuracy')
    else:
        plt.ylabel('Error')
    plt.legend()
    plt.show()


def comp_acc_vs_noises():

    sigma_sqr = np.arange(0.1, 0.6, 0.1)

    plot_acc = False
    add_noise = True


if __name__ == '__main__':

    comp_acc_vs_sizes()