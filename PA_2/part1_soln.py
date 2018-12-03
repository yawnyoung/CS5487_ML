"""
Part1 solutions

author: Yajue Yang
"""

from PA_2.data_processing import *
from PA_2.clustering_alg import k_means, EM_GMM, mean_shift_clustering
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

POINT_COLOR = ['r', 'g', 'b', 'y']


def plot_k_means_clustering(x, z, means, fig_name):

    num_data = x.shape[1]

    for i in range(num_data):

        c_idx = np.argmax(z[:, i])

        plt.plot(x[0, i], x[1, i], '.', color=POINT_COLOR[c_idx])

    plt.plot(means[0, :], means[1, :], 'ko')
    plt.title(fig_name)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.show()


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill=False, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_gmm_clustering(x, means, cov, z, fig_name):

    num_data = x.shape[1]

    for i in range(num_data):

        c_idx = np.argmax(z[:, i])

        plt.plot(x[0, i], x[1, i], '.', color=POINT_COLOR[c_idx])

    plt.plot(means[0, :], means[1, :], 'ko')
    plt.title(fig_name)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    for i in range(4):
        plot_cov_ellipse(cov[:, :, i], means[:, i])

    plt.show()


def test_k_means():
    x, y = load_synthetic_data('C')

    # init_mean = np.array([[-3, 3, 4, -2], [3, 4, -3, -2]])
    # init_mean = np.random.rand(2, 4) * 10
    init_mean = np.random.uniform(-15, 15, 2 * 4).reshape((2, 4))
    print(init_mean)

    epsilon = np.array([0.001, 0.001, 0.001, 0.001])

    curr_mean, z = k_means(x, init_mean, epsilon)

    plot_k_means_clustering(x, z, curr_mean, 'dataC',)

    # print(curr_mean)
    # print(z.shape)


def test_em_gmm():

    data_name = 'C'

    x, y = load_synthetic_data(data_name)

    dim_param = x.shape[0]
    init_pis = np.array([0.25, 0.25, 0.25, 0.25])
    # init_mean = np.array([[-3, 3, 4, -2], [3, 4, -3, -2]])
    init_mean = np.random.uniform(-15, 15, 2 * 4).reshape((2, 4))

    # epsilon = np.array([0.001, 0.001, 0.001, 0.001])
    #
    # init_mean, z = k_means(x, init_mean, epsilon)

    print(init_mean)
    init_cov = np.ndarray((dim_param, dim_param, 4))
    for i in range(4):
        init_cov[:, :, i] = np.identity(dim_param) * 0.1

    pis, means, cov, z = EM_GMM(x, init_pis, init_mean, init_cov, 0.001)

    plot_gmm_clustering(x, means, cov, z, 'data' + data_name)


def test_mean_shift():

    data_name = 'A'

    x, y = load_synthetic_data(data_name)

    h = 0.5
    ct = 0.001
    pr_min = 0.1
    x_cnvg, peaks, z = mean_shift_clustering(x, h, ct, pr_min)

    colors = cm.rainbow(np.linspace(0, 1, len(peaks)))
    num_data = x.shape[1]
    for i in range(num_data):
        plt.plot(x[0, i], x[1, i], '.', color=colors[int(z[i])])

    plt.title('data' + data_name)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()


if __name__ == '__main__':

    # test_k_means()

    # test_em_gmm()
    test_mean_shift()

    # colors = cm.rainbow(np.linspace(0, 1, 10))
    #
    # plt.plot(1, 2, '.', )