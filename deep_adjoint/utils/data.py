import numpy as np
import glob
import pickle
import random
import os
import torch
import h5py
from torch.utils.data import Dataset
from .utils import split_idx
from .scaler import DataScaler
from sklearn.preprocessing import MinMaxScaler



class SOMA_PCA_Data(Dataset):
    def __init__(self, path, mode, hist_len=1) -> None:
        super().__init__()
        f = h5py.File(path, 'r')
        data = f['PCA_data'][:]
        self.var_idx = [3, 6, 10, 14, 15] 
        self.scaler = MinMaxScaler()

        train_idx, val_idx, test_idx = split_idx(data.shape[0])
        self.scaler.fit(data[train_idx][..., self.var_idx].reshape(len(train_idx), -1)) # fit a minmax scaler using the training data

        if mode == 'train':
            self.data = data[train_idx][..., self.var_idx].reshape(len(train_idx), -1)
        elif mode == 'val':
            self.data = data[val_idx][..., self.var_idx].reshape(len(val_idx), -1)
        elif mode == 'test':
            self.data = data[test_idx][..., self.var_idx].reshape(len(test_idx), -1)
        else:
            raise NameError(f"Mode name {mode} not found!")

        self.data = self.scaler.transform(self.data) 
        
        self.x, self.y = self.prepare_data(hist_len=hist_len)
        if self.x.shape[1] == 1:
            self.x = np.squeeze(self.x)

    def prepare_data(self, hist_len=1):
        x = []
        y = []
        for i in range(self.data.shape[0] - hist_len):
            x.append(self.data[i:i+hist_len])
            y.append(self.data[i+hist_len])
        
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index]).float()
        y = torch.from_numpy(self.y[index]).float()
        return x, y



class GlacierData(Dataset):
    def __init__(self, path, mode='train', portion='u'):
        super().__init__()
        # DIR = os.path.dirname(os.path.abspath(__file__))
        # path = os.path.join(DIR, path)
        data = np.load(path) # use the first half for training
        self.portion = portion
        self.inputs = data['inputs']
        self.uout = data['uout']
        self.jac_beta = data['jac_beta']
        self.jac_u = data['jac_u']

        train_idx, val_idx, test_idx = split_idx(self.inputs.shape[0])
        if mode == 'train':
            self.inputs = self.inputs[train_idx]
            self.uout = self.uout[train_idx]
            self.jac_beta = self.jac_beta[train_idx]
            self.jac_u = self.jac_u[train_idx]

        elif mode == 'val':
            self.inputs = self.inputs[val_idx]
            self.uout = self.uout[val_idx]
            self.jac_beta = self.jac_beta[val_idx]
            self.jac_u = self.jac_u[val_idx]
        elif mode == 'test':
            self.inputs = self.inputs[test_idx]
            self.uout = self.uout[test_idx]
            self.jac_beta = self.jac_beta[test_idx]
            self.jac_u = self.jac_u[test_idx]
    def __len__(self):
        return self.inputs.shape[0]
    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.uout[idx]).float()
        if self.portion == 'u':
            adj = torch.from_numpy(self.jac_u)[idx].float()
        elif self.portion == 'p':
            adj = torch.from_numpy(self.jac_beta)[idx].float()
        else:
            raise Exception(f"Portion type {self.portion} not recognized!")
        return x, (y, adj)
    
class SOMAdata(Dataset):
    def __init__(self, path, mode, gpu_id, time_steps_per_forward=30, transform=True, train_noise=False, loss_mask=False):
        '''path: the hd5f file path, can be relative path
        mode: ['trian', 'val', 'test']
        '''
        super(SOMAdata, self).__init__()
        self.mode = mode
        self.train_noise = train_noise

        DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(DIR, path)
        self.data = h5py.File(data_path, 'r')
        keys = list(self.data.keys())

        random.Random(0).shuffle(keys)
        TRAIN_SIZE = int(0.6 * len(keys))
        TEST_SIZE = int(0.1 * len(keys))

        self.gpu_id = gpu_id
        self.time_steps_per_forward = time_steps_per_forward
        
        # specify the range of state variables in order
        '''
        Layer thickness: [4.539446, 13.05347]
        Salinity: [34.01481, 34.24358].
        Temperature: [5.144762, 18.84177]
        Meridional Velocity: [3.82e-8, 0.906503]
        Zonal Velocity: [6.95e-9, 1.640676]
        '''
        data_min = np.array([34.01481, 5.144762, 4.539446, 3.82e-8, 6.95e-9, 200]) 
        data_max = np.array([34.24358, 18.84177, 13.05347, 0.906503, 1.640676, 2000])

        self.scaler = DataScaler(data_min=data_min, data_max=data_max)
        self.transform = transform


        with open('/global/homes/y/yixuans/DeepAdjoint/tmp/SOMA_mask.pkl', 'rb') as f:
                mask = pickle.load(f)
            
        self.mask1 = mask['mask1']
        self.mask2 = mask['mask2']
        self.mask = np.logical_or(self.mask1, self.mask2)[0,0,:,:,0]

        # create a mask for loss calculation
        if loss_mask:

            self.loss_mask = np.logical_or(self.mask1, self.mask2)[0,0,:,:,0] # mask only in x,y plane thus size of [100, 100] this will broadcast in element wise product
            self.loss_mask = np.array(~self.loss_mask, dtype=int) # True - 0; False - 1
            self.loss_mask = torch.from_numpy(self.loss_mask).float()

        if mode == 'train':
            self.keys = keys[:TRAIN_SIZE]
        elif mode == 'val':
            self.keys = keys[TRAIN_SIZE: TRAIN_SIZE + TEST_SIZE]
        elif mode == 'test':
            self.keys = keys[-TEST_SIZE:]
            print("Test set keys", self.keys)
        else:
            raise Exception(f'Invalid mode: {mode}, please select from "train", "val", and "test".')

    def preprocess(self, x, y):
        '''Prepare data as the input-output pair for a single forward run
        x has the shape of (60, 100, 100, 5)
        the goal is to first move the ch axis to the second -> (5, 60, 100, 100)
        then create input output pair where the input shape is (5, 60, 100, 100) and the output shape is (16, 60, 100, 100)
        idx 14 is the varying parameter for the input.

        '''
        assert len(x.shape) == 4, "Incorrect data shape!"
        var_idx = [7, 8, 11, 14, 15, -1] #[3, 6, 10, 14, 15] # needs adjusting for the daily averaged datasets [7, 8, 11, 14, 15] 

        x = x[..., var_idx]
        y = y[..., var_idx]

        if self.transform:
            x = self.scaler.transform(x)
            y = self.scaler.transform(y)

        bc_mask = np.broadcast_to(self.mask[np.newaxis, ..., np.newaxis], x.shape)
        x[bc_mask] = 0
        y[bc_mask] = 0


        x_in = np.transpose(x, axes=[3, 0, 1, 2])
        x_out = np.transpose(y, axes=[3, 0, 1, 2])[:-1, ...] # remove the parameter dim


        if self.train_noise and self.mode != 'test':
            noise = np.random.normal(loc=0.0, scale=3e-4, size=x_in.shape)  #http://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf
            x_in = x_in + noise
        return (x_in, x_out)

    def __len__(self):
        return len(self.keys) * (self.time_steps_per_forward - 1) # b/c n  time steps can create n-1 input-output pairs
        
    def __getitem__(self, index):
        # get the key idx 
        key_idx = int(index / (self.time_steps_per_forward - 1))
        in_group_idx = index % (self.time_steps_per_forward - 1)
        data_x = self.data[self.keys[key_idx]][in_group_idx]
        data_y = self.data[self.keys[key_idx]][in_group_idx + 1]
        x, y = self.preprocess(data_x, data_y)
        assert not np.any(np.isnan(x)) and not np.any(np.isnan(y)), "Data contains NaNs!!!"
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class MultiStepData(Dataset):
    def __init__(self, data_name='burgers', path='./deep_adjoint/Data/mixed_nu/', mode='Train'):
        super(MultiStepData, self).__init__()

        if mode == 'val':
            path = path + 'val/'
        elif mode == 'test':
            pass

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
