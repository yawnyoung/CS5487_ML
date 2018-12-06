"""
Project data processing

author: Yajue Yang
"""

import numpy as np


def load_data():
    file_name = 'wine_data.txt'

    targets = []
    features = []

    with open(file_name) as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.split(',')
            targets.append(int(line[0]))
            features.append([float(f) for f in line[1:]])

    return features, targets


def targets_stat(targets):
    num_c1 = 0
    num_c2 = 0
    num_c3 = 0

    for t in targets:
        if t == 1:
            num_c1 += 1
        elif t == 2:
            num_c2 += 1
        else:
            num_c3 += 1

    return np.array([num_c1, num_c2, num_c3])


def load_train_test_data(train_ratio=0.9, shuffle=False):

    features, targets = load_data()

    dim_feat = len(features[0])

    hist_classes = targets_stat(targets)
    # print('histogram of classes: ', hist_classes)

    if shuffle:
        for k in range(3):
            class_x_y = np.ndarray((hist_classes[k], dim_feat + 1))
            start_idx = 0
            if k == 1:
                start_idx = hist_classes[k - 1]
            if k == 2:
                start_idx = hist_classes[k - 1] + hist_classes[k - 2]
            for i in range(hist_classes[k]):
                class_x_y[i, :-1] = np.array(features[start_idx+i])
                class_x_y[i, -1] = targets[start_idx + i]

            np.random.shuffle(class_x_y)
            for i in range(hist_classes[k]):
                features[start_idx + i] = class_x_y[i, :-1].tolist()
                targets[start_idx + i] = int(class_x_y[i, -1])

    num_train = np.array(hist_classes * train_ratio, dtype=int)
    # print('number of train: ', num_train)

    train_tgt = []
    train_feat = []
    test_tgt = []
    test_feat = []
    for i in range(3):
        train_start_idx = 0
        for j in range(i):
            train_start_idx += hist_classes[j]
        train_end_idx = train_start_idx + num_train[i]

        train_tgt += targets[train_start_idx:train_end_idx]
        test_tgt += targets[train_end_idx:train_end_idx+hist_classes[i] - num_train[i]]

        train_feat += features[train_start_idx:train_end_idx]
        test_feat += features[train_end_idx:train_end_idx+hist_classes[i] - num_train[i]]

    train_feat_arr = np.array(train_feat)
    train_feat_mean = np.average(train_feat_arr, axis=0)
    train_feat_std = np.std(train_feat_arr, axis=0)

    # whiten train features
    train_feat_arr = (train_feat_arr - train_feat_mean) / train_feat_std
    train_feat = train_feat_arr.tolist()

    # whiten test features
    test_feat_arr = (np.array(test_feat) - train_feat_mean) / train_feat_std
    test_feat = test_feat_arr.tolist()

    return (train_tgt, train_feat), (test_tgt, test_feat)


def add_gaussian_noises(data, sigma_sqr):

    y = data[0].copy()
    x = data[1].copy()

    num_data = len(y)
    dim_param = len(x[0])

    # cov = np.identity(dim_param) * sigma_sqr
    cov = np.identity(dim_param)

    noise_data_idx = np.random.randint(0, num_data, int(0.5 * num_data)).tolist()

    for i in noise_data_idx:
        # print('before: ', x[i])
        noise = np.random.multivariate_normal(np.random.uniform(low=-1, high=1, size=dim_param).reshape(dim_param)*sigma_sqr, cov, 1).flatten()
        # print('noise: ', noise)
        x_arr = np.array(x[i]) + noise
        x_arr = x_arr.flatten()
        # print('after x_arr: ', x_arr)
        x[i] = x_arr.tolist().copy()
        # print('after: ', x[i])

    return (y, x)


if __name__ == '__main__':

    # load_data()

    train_data, test_data = load_train_test_data(shuffle=True)

    # print('train data:\n', train_data)
    # print('test data:\n', test_data)

    # train_data = add_gaussian_noises(train_data, 0.1)