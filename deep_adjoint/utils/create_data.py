import pickle

import h5py
import numpy as np
from tqdm import tqdm
from utils import split_idx


def get_time_series(
    data,
    hist_len,
    horizon,
    var_id_x,
    var_id_y,
    vertical_start=0,
    vertical_end=3,
    if_two_d=False,
):
    """Get the time series data from the given data
    data: np.array of shape (n, 60, 100, 100, var_len)
    """
    with open(
        "/global/homes/y/yixuans/DeepAdjoint/tmp/SOMA_mask.pkl", "rb"
    ) as f:
        mask = pickle.load(f)

    mask1 = mask["mask1"]
    mask2 = mask["mask2"]
    mask = np.logical_or(mask1, mask2)[
        0:1, vertical_start:vertical_end, :, :, 0:1
    ]
    x_ = []
    y_ = []

    # if isinstance(vertical_level, int):
    #     print('Horizontal slices, 2D in space...')

    for i in range(data.shape[0] - hist_len - horizon + 1):
        x = data[
            i : i + hist_len, vertical_start:vertical_end, ..., var_id_x[0:]
        ]
        y = data[
            i + hist_len : i + hist_len + horizon,
            vertical_start:vertical_end,
            ...,
            var_id_y[0:],
        ]

        # if hist_len == 1 and horizon == 1:
        #     assert x.shape == (1, 60, 100, 100, len(var_id_x))
        #     assert y.shape == (1, 60, 100, 100, len(var_id_y))

        mask_x = np.broadcast_to(mask, x.shape)
        mask_y = np.broadcast_to(mask, y.shape)
        x[mask_x] = 0
        y[mask_y] = 0
        x_.append(x)
        y_.append(y)

    x = np.stack(x_, axis=0)
    y = np.stack(y_, axis=0)

    if if_two_d:
        x = x.reshape(-1, 100, 100, x.shape[-1])
        y = y.reshape(-1, 100, 100, y.shape[-1])

    return x, y


def make_h5data(data_path, dataset_name, data_f, keys, mode):
    for i, k in tqdm(enumerate(keys)):
        x, y = get_time_series(
            data_f[k],
            1,
            10,
            [8, -1],
            [8],
        )  # Adjust as per your requirements

        # Open the file in append mode or create it if it's the first iteration
        file_path = data_path + f"{dataset_name}_{mode}.h5"
        with h5py.File(file_path, "a") as new_f:
            if i == 0:  # First dataset, create the file and datasets
                new_f.create_dataset(
                    "x",
                    shape=(0,) + x.shape[1:],
                    maxshape=(None,) + x.shape[1:],
                    dtype=x.dtype,
                )
                new_f.create_dataset(
                    "y",
                    shape=(0,) + y.shape[1:],
                    maxshape=(None,) + y.shape[1:],
                    dtype=y.dtype,
                )

            dset_x = new_f["x"]
            dset_y = new_f["y"]

            # Determine the current size of the datasets
            current_size_x = dset_x.shape[0]
            current_size_y = dset_y.shape[0]

            # Resize the datasets to accommodate the new data
            dset_x.resize(current_size_x + x.shape[0], axis=0)
            dset_y.resize(current_size_y + y.shape[0], axis=0)

            # Append the new data
            dset_x[current_size_x:] = x
            dset_y[current_size_y:] = y


def create_datasets(dataset_name, test_size=0.1):
    SCRACTH = "/pscratch/sd/y/yixuans/"
    f = h5py.File(SCRACTH + "datatset/SOMA/thedataset-GM-dayAvg-2.hdf5", "r")
    keys = list(f.keys())
    train_key, val_key, test_key = split_idx(len(f.keys()), test_size)
    train_key = [keys[i] for i in train_key]
    val_key = [keys[i] for i in val_key]
    test_key = [keys[i] for i in test_key]
    make_h5data(SCRACTH, dataset_name, f, train_key, "train")
    make_h5data(SCRACTH, dataset_name, f, val_key, "val")
    make_h5data(SCRACTH, dataset_name, f, test_key, "test")


if __name__ == "__main__":
    create_datasets("dyffusion_temp_depth-3_horizon-10")
