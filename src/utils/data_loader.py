import numpy as np

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