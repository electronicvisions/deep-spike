'''
Some functions useful in the code.
In particular to load the data with scikit learn.

'''

import numpy as np
from sklearn.datasets import load_digits

def load_small_digits(train_prop,n_class):
    '''
    Load the data from the scikit learn dataset

    :param train_prop: proportion of samples in the testing set<
    :param n_class: number of different digits
    :return:
    '''

    # Load the 8 by 8 digit dataset
    data = load_digits(n_class)
    N_images = data.target.size
    N_train = int(N_images * train_prop)
    N_test = N_images - N_train

    x_train = data.data[:N_train,:]
    x_test = data.data[N_train:,:]

    class_train = data.target[:N_train]
    class_test = data.target[N_train:]

    z_train = np.zeros((N_train,n_class))
    z_train[np.arange(N_train),class_train] = 1
    z_test = np.zeros((N_test,n_class))
    z_test[np.arange(N_test),class_test] = 1

    return x_train,x_test,z_train,z_test