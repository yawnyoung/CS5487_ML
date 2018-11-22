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


def load_train_test_data(train_ratio=0.9):
    features, targets = load_data()

    hist_classes = targets_stat(targets)
    # print('histogram of classes: ', hist_classes)

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

    return (train_tgt, train_feat), (test_tgt, test_feat)


if __name__ == '__main__':

    train_data, test_data = load_train_test_data()

    print('train data:\n', train_data)
    print('test data:\n', test_data)