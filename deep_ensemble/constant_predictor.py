import torch
import numpy as np
from torch.utils.data import DataLoader
from data import SOMAdata

valset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5", "val", transform=True
    )

valLoader = DataLoader(valset, batch_size=valset.__len__())

for x, y in valLoader:
    print(x.shape)
    print(y.shape)
    y = y.numpy()
    m = np.mean(y, axis=(0,2,3))
    m = m.reshape(1, 5, 1, 1)

    loss = np.mean((y-m)**2)
    print(m.shape)
    print(loss)