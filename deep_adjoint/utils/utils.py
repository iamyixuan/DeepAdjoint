import numpy as np
from sklearn.decomposition import PCA

def split_idx(idx_len):
    rs = np.random.RandomState(0)
    idx = rs.permutation(idx_len)
    train_size = int(.5 * idx_len)
    test_size = int(.25 * idx_len)

    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + test_size]
    test_idx = idx[-test_size:]
    return train_idx, val_idx, test_idx


class ChannelScaler:
    def __init__(self) -> None:
        pass
    def centering(self, data, axis=1):
        m_ = np.mean(data, axis=axis, keepdims=True)
        std_ = np.std(data, axis=axis, keepdims=True)
        mask = std_ == 0

        self.m_ = m_
        self.std_ = std_

        std_[mask] = 1
        return (data - m_) / std_
    def transform(self, x):
        return (x - self.m_) / self.std_
    
    def inverse_transform(self, x):
        return x * self.std_ + self.m_

def channel_pca(data):
    batch = data.shape[0]
    ch = data.shape[-1]
    data_pca = data.reshape(batch, -1, ch)
    PCAs = []
    for i in range(ch):
        pca = PCA(n_components=50)
        pca.fit(data_pca[...,i])
        PCAs.append(pca)

    return PCAs

