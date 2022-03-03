import numpy as np

def g(x, a):
    y = np.append(1, x)
    return a.T @ y.T


def minste_kvadraters(test_data, train_data, feature_idx):

    """Kalkulerer feilraten for den beste egenskapskombinasjon ved bruk av minste kvadraters metode
    
    Arguments:
    
    test_data => input test data
    
    train_data => input train data
    
    feature_idx => Indeks for den beste egenskapskombinasjonen
    
    Return value:
    err_rate => Klassifiserer feilraten for en gitt egenskapskombinasjon
    """

    test_objects = test_data[:, 1:]
    test_objects = test_objects[:, feature_idx]
    train_objects = train_data[:, 1:]
    train_objects = train_objects[:, feature_idx]

    idx1 = train_data[:, 0] == 1
    idx2 = ~idx1

    Y = np.c_[np.ones((train_objects.shape[0], 1)), train_objects]
    b = idx1 + (-1) * idx2

    a = np.linalg.pinv(Y.T @ Y) @ Y.T @ b

    err_rate = 0
    for i in range(test_objects.shape[0]):
        g_ = g(test_objects[i], a)
        if g_ > 0:
            class_error = test_data[i, 0] != 1
        else:
            class_error = test_data[i, 0] != 2
        err_rate += class_error
    err_rate /= test_objects.shape[0]

    return err_rate