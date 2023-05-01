import numpy as np
import glob
import pickle

def combine_burgers_data(folderPath):
    '''combining all the datasets into one'''
    files = glob.glob(folderPath + '*.pkl')
    x_ = []
    y_ = []
    adj_ = []
    for f in files:
        with open(f, 'rb') as g:
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
    '''
    Load the generated burgers data with different nu
        The solution has the shape [NX, NT]. 
        Add Nu as the last element to the input. 
        The output is only the solution with one step ahead
    '''
    sol = np.array(data[0])
    adj = np.array(data[1])
    Nu = np.array([data[2]])
    x = []
    y = []
    for t in range(sol.shape[0] - 1): # the first axis being the time.
        x_ = sol[t, :]
        y_ = sol[t+1, :]
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
        x_= x[:5000]
        y_ = y[:5000]
        adj_ = adj[:5000]

        x_test = x[5000:]
        y_test = y[5000:]
        adj_test = adj[5000:]

    idx = rd.permutation(x_.shape[0])
    train_len = int(0.8*len(idx))
    train_idx = idx[:train_len]
    val_idx = idx[train_len:]

    x_train = x_[train_idx]
    y_train = y_[train_idx]
    adj_train = adj_[train_idx]

    x_val = x_[val_idx]
    y_val = y_[val_idx]
    adj_val = adj_[val_idx]

    train = {'x': x_train, 'y': y_train, 'adj': adj_train}
    val = {'x': x_val, 'y': y_val, 'adj': adj_val}
    test = {'x': x_test, 'y': y_test, 'adj': adj_test}
    return train, val, test


if __name__ == "__main__":
    data = np.load('../data/vary_A_glen.dat.npz')
    inputs = data['inputs']
    uout = data['uout']
    jrav = data['jrav']
    train, val, test = split_data(inputs, uout, jrav)
    print(train['x'].shape)
    print(train['y'].shape)

    print(train['x'][:5])
    print(train['y'][:5])