import numpy as np


def get_stratified_subsample(X, y, percent):

    x_sample, y_sample = [], []

    for label in np.unique(y):

        x_label, y_label = X[y == label], y[y == label]

        n_sample = int(percent * x_label.shape[0])
        idx_sample = np.random.random_integers(0, n_sample, n_sample)

        x_sample.append(x_label[idx_sample])
        y_sample.append(y_label[idx_sample])

    X_out, y_out = x_sample[0], y_sample[0]

    for x, y in zip(x_sample[1:], y_sample[1:]):

        X_out = np.concatenate((X_out, x), axis=0)
        y_out = np.concatenate((y_out, y), axis=0)

    return X_out, y_out
