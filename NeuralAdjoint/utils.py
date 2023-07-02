import torch
import glob
import pickle
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise Exception(
            "Specified activation function {} not found.".format(activation)
        )


def combine_burgers_data(folderPath):
    """combining all the datasets into one"""
    files = glob.glob(folderPath + "*.pkl")
    x_ = []
    y_ = []
    adj_ = []
    for f in files:
        with open(f, "rb") as g:
            tmp = pickle.load(g)
        x, y, adj = load_burgers_data(tmp)
        x_.append(x)
        y_.append(y)
        adj_.append(adj)

    x_ = np.concatenate(x_, axis=0)
    y_ = np.concatenate(y_, axis=0)
    adj_ = np.concatenate(adj_, axis=0)
    return x_, y_, adj_


def load_burgers_data(data):
    """
    Load the generated burgers data with different nu
        The solution has the shape [NX, NT].
        Add Nu as the last element to the input.
        The output is only the solution with one step ahead
    """
    sol = np.array(data[0])
    adj = np.array(data[1])
    Nu = np.array([data[2]])
    x = []
    y = []
    for t in range(sol.shape[0] - 1):  # the first axis being the time.
        x_ = sol[t, :]
        y_ = sol[t + 1, :]
        x_ = np.concatenate([x_, Nu], axis=0)
        x.append(x_)
        y.append(y_)
    x = np.array(x)
    y = np.array(y)
    return x, y, adj


def split_data(x, y, adj, shuffle_all=False):
    # use the first 100 cases (first 50*100 indices for traning and val) and the rest for testing
    data_len = x.shape[0]
    rd = np.random.RandomState(0)

    train_len = int(0.8 * data_len)
    if shuffle_all:
        idx_sh = rd.permutation(x.shape[0])
        x_ = x[idx_sh[:train_len]]
        y_ = y[idx_sh[:train_len]]
        adj_ = adj[idx_sh[:train_len]]

        x_test = x[idx_sh[train_len:]]
        y_test = y[idx_sh[train_len:]]
        adj_test = adj[idx_sh[train_len:]]
    else:
        x_ = x[idx_sh[:train_len]]
        y_ = y[idx_sh[:train_len]]
        adj_ = adj[idx_sh[:train_len]]

        x_test = x[idx_sh[train_len:]]
        y_test = y[idx_sh[train_len:]]
        adj_test = adj[idx_sh[train_len:]]

    idx = rd.permutation(x_.shape[0])
    train_len = int(0.8 * len(idx))
    train_idx = idx[:train_len]
    val_idx = idx[train_len:]

    x_train = x_[train_idx]
    y_train = y_[train_idx]
    adj_train = adj_[train_idx]

    x_val = x_[val_idx]
    y_val = y_[val_idx]
    adj_val = adj_[val_idx]

    train = {"x": x_train, "y": y_train, "adj": adj_train}
    val = {"x": x_val, "y": y_val, "adj": adj_val}
    test = {"x": x_test, "y": y_test, "adj": adj_test}
    return train, val, test


class BurgersDataset(Dataset):
    def __init__(self, x, y, device):
        super(BurgersDataset, self).__init__()
        self.x = x
        self.y = y
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]).float().to(
            self.device
        ), torch.from_numpy(self.y[index]).float().to(self.device)
