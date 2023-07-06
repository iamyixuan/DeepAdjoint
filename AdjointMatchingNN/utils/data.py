import numpy as np
import glob
import pickle
import random
import os
import torch
import h5py
from torch.utils.data import Dataset


class SOMAdata(Dataset):
    def __init__(self, path, mode, device, time_steps_per_forward=3):
        '''path: the hd5f file path, can be relative path
        mode: ['trian', 'val', 'test']
        '''
        super(SOMAdata, self).__init__()
        DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(DIR, path)
        self.data = h5py.File(data_path, 'r')
        keys = list(self.data.keys())
        keys.remove('forward_233')
        random.Random(0).shuffle(keys)
        TRAIN_SIZE = int(0.8 * len(keys))
        TEST_SIZE = int(0.1 * len(keys))
        self.device = device
        self.time_steps_per_forward = time_steps_per_forward

        if mode == 'train':
            self.keys = keys[:TRAIN_SIZE]
        elif mode == 'val':
            self.keys = keys[TRAIN_SIZE: TRAIN_SIZE + TEST_SIZE]
        elif mode == 'test':
            self.keys = keys[-TEST_SIZE:]
        else:
            raise Exception(f'Invalid mode: {mode}, please select from "train", "val", and "test".')

    def preprocess(self, x):
        '''Prepare data as the input-output pair for a single forward run
        x has the shape of (3, 185, 309, 60, 15)
        the goal is to first move the ch axis to the second -> (3, 15, 185, 309, 60)
        then create input output pair where the input shape is (1, 15, 185, 309, 60, 15) and the output shape is (1, 14, 185, 309, 60)
        idx 14 is the varying parameter for the input.
        '''
        x = np.transpose(x, axes=[0, 4, 1, 2, 3])
        x_in = x[:-1]
        x_out = x[1:, :-1, ...] 
        return (x_in, x_out)

    def __len__(self):
        return int(len(self.keys) * (self.time_steps_per_forward - 1))
        
    def __getitem__(self, index):
        # get the key idx 
        key_idx = int(index / (self.time_steps_per_forward - 1))
        in_group_idx = index % (self.time_steps_per_forward - 1)
        data = self.data[self.keys[key_idx]]['month_0'][...]
        x, y = self.preprocess(data)
        return torch.from_numpy(x[in_group_idx]).float().to(self.device), torch.from_numpy(y[in_group_idx]).float().to(self.device)


class MultiStepData(Dataset):
    def __init__(self, data_name='burgers', path='./AdjointMatchingNN/Data/mixed_nu/'):
        super(MultiStepData, self).__init__()
        if data_name == 'burgers':
            data = MultiStepBurgers(path)
        self.sol = data.sol
        self.adj = data.adj

    def __len__(self):
        return self.sol.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.sol[idx, 0, ...]).float(), (
            torch.from_numpy(self.sol[idx, 1:, :-1]).float(),
            torch.from_numpy(self.adj[idx, :, :, :-1]).float(),
        )

class MultiStepBurgers:
    def __init__(self, path, aug_state=True) -> None:
        self.path = path
        self.aug_state = aug_state
        self.sol, self.adj = self.combineData()
    def combineData(self):
        files = glob.glob(self.path + "*.pkl")
        sol = []
        adj = []
        for f in files:
            with open(f, 'rb') as f:
                tmp = pickle.load(f)
                nu = np.repeat(tmp[2], tmp[0].shape[0]).reshape(-1, 1)
                if self.aug_state:# augment the state variables with the external model parameter nu
                    sol_tmp = np.concatenate([tmp[0], nu], axis=1) # (201, 128) -> (201, 129), the lasting being \nu.
                else:
                    sol_tmp = tmp[0]
                sol.append(sol_tmp)
                adj.append(tmp[1])
        return np.array(sol), np.array(adj)

        

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

    if shuffle_all:
        idx_sh = rd.permutation(x.shape[0])
        x_ = x[idx_sh[:5000]]
        y_ = y[idx_sh[:5000]]
        adj_ = adj[idx_sh[:5000]]

        x_test = x[idx_sh[5000:]]
        y_test = y[idx_sh[5000:]]
        adj_test = adj[idx_sh[5000:]]
    else:
        x_ = x[:5000]
        y_ = y[:5000]
        adj_ = adj[:5000]

        x_test = x[5000:]
        y_test = y[5000:]
        adj_test = adj[5000:]

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


if __name__ == "__main__":
    data = np.load("../data/vary_A_glen.dat.npz")
    inputs = data["inputs"]
    uout = data["uout"]
    jrav = data["jrav"]
    train, val, test = split_data(inputs, uout, jrav)
    print(train["x"].shape)
    print(train["y"].shape)

    print(train["x"][:5])
    print(train["y"][:5])
