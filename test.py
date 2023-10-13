import numpy as np
import h5py
from scipy.io import loadmat


def load_data(size=5):

    f = loadmat('data/traffic_dataset.mat')
    # train_hr = len(f['tra_X_tr'][0])
    train_x = [i[:size] for i in f['tra_X_tr'][0]]
    train_y = [i[:size] for i in f['tra_Y_tr'][0]]

    test_x = [i[:size] for i in f['tra_X_te'][0]]
    test_y = [i[:size] for i in f['tra_Y_te'][0]]

    adj = f['tra_adj_mat'][:size, :size]

    return train_x, train_y, test_x, test_y, adj
    

train_x, train_y, test_x, test_y, adj = load_data()

