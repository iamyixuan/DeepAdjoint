import os
import pickle
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def add_noise(data, scale_list):
    loc = np.zeros(data.shape)
    scales = np.ones_like(data)
    for i in range(len(scale_list)):
        scales[..., i] = scale_list[i] * scales[..., i]
    data = data + np.random.normal(loc, scales, data.shape)
    return data


class MinMaxScaler:
    def __init__(self, data, min_=0, max_=1) -> None:
        data = data.astype(np.float64)
        self.data_min = np.min(data)
        self.data_max = np.max(data)
        self.min_ = min_
        self.max_ = max_

    def transform(self, x):
        d_diff = self.data_max - self.data_min
        mask = d_diff == 0
        d_diff[mask] = 1
        s_diff = self.max_ - self.min_

        res = (x - self.data_min) / d_diff * s_diff + self.min_
        return res.astype(np.float32)

    def inverse_transform(self, x):
        d_diff = self.data_max - self.data_min
        s_diff = self.max_ - self.min_
        return (x - self.min_) / s_diff * d_diff + self.data_min


class ChannelMinMaxScaler(MinMaxScaler):
    def __init__(self, data, axis_apply, min_=0, max_=1) -> None:
        super().__init__(data, min_, max_)
        data = data.astype(np.float64)
        self.data_min = np.nanmin(data, axis=axis_apply, keepdims=True)
        self.data_max = np.nanmax(data, axis=axis_apply, keepdims=True)


class SimpleDataset(Dataset):
    def __init__(self, file_path):
        super(SimpleDataset, self)
        self.data = h5py.File(file_path, "r")
        maxs = np.array([18.84177, 13.05347, 0.906503, 1.640676, 2000])
        mins = np.array([5.144762, 4.539446, 3.82e-8, 6.95e-9, 200])

        self.maxs = maxs[None, None, :]
        self.mins = mins[None, None, :]

    def __len__(self):
        return self.data["x"].shape[0]

    def __getitem__(self, index):
        x = self.data["x"][index, ...]
        x = self.transform(x)
        x = np.transpose(x, axes=[2, 0, 1])

        y = self.data["y"][index, ...]
        y = self.transform(y)
        y = np.transpose(y, axes=[2, 0, 1])
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def transform(self, x):
        if x.shape[-1] == 4:
            return (x - self.mins[..., :-1]) / (self.maxs - self.mins)[
                ..., :-1
            ]
        else:
            return (x - self.mins) / (self.maxs - self.mins)

    def inverse_transform(self, x):
        if x.shape[-1] == 4:
            return x * (self.maxs - self.mins)[..., :-1] + self.mins[..., :-1]
        else:
            return x * (self.maxs - self.mins) + self.mins


class DataScaler:
    """
    Layer thickness: [4.539446, 13.05347]
    Salinity: [34.01481, 34.24358].
    Temperature: [5.144762, 18.84177]
    Meridional Velocity: [3.82e-8, 0.906503]
    Zonal Velocity: [6.95e-9, 1.640676]
    """

    def __init__(self, data_min, data_max, min_=0, max_=1) -> None:
        # super().__init__( min_, max_)
        self.data_min = data_min.reshape(1, 1, 6)
        self.data_max = data_max.reshape(1, 1, 6)
        self.min_ = min_
        self.max_ = max_

    def transform(self, x):
        d_diff = self.data_max - self.data_min
        mask = d_diff == 0
        d_diff[mask] = 1
        s_diff = self.max_ - self.min_

        res = (x - self.data_min) / d_diff * s_diff + self.min_
        return res.astype(np.float32)

    def inverse_transform(self, x):
        d_diff = self.data_max - self.data_min
        s_diff = self.max_ - self.min_
        return (x - self.min_) / s_diff * d_diff + self.data_min


class SOMAdata(Dataset):
    def __init__(
        self,
        path,
        mode,
        time_steps_per_forward=30,
        transform=False,
        x_noise=False,
        y_noise=False,
    ):
        """path: the hd5f file path, can be relative path
        mode: ['trian', 'val', 'test']
        """
        super(SOMAdata, self).__init__()
        self.mode = mode
        self.x_noise = x_noise
        self.y_noise = y_noise

        self.data = h5py.File(path, "r")
        keys = list(self.data.keys())

        random.Random(0).shuffle(keys)
        TRAIN_SIZE = int(0.6 * len(keys))
        TEST_SIZE = int(0.1 * len(keys))

        self.time_steps_per_forward = time_steps_per_forward

        # data order [layer thickness, salinity, temp, meri v, zonal v]
        data_min = np.array(
            [4.539446, 34.01481, 5.144762, 3.82e-8, 6.95e-9, 200]
        )
        data_max = np.array(
            [13.05347, 34.24358, 18.84177, 0.906503, 1.640676, 2000]
        )

        self.scaler = DataScaler(data_min=data_min, data_max=data_max)
        self.transform = transform

        # print(sample_data.shape)
        with open("../tmp/SOMA_mask.pkl", "rb") as f:
            mask = pickle.load(f)

        self.mask1 = mask["mask1"]
        self.mask2 = mask["mask2"]
        self.mask = np.logical_or(self.mask1, self.mask2)[0, 0, :, :, 0]

        if mode == "train":
            self.keys = keys[:TRAIN_SIZE]
        elif mode == "val":
            self.keys = keys[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]
        elif mode == "test":
            self.keys = keys[-TEST_SIZE:]
            print("Test set keys", self.keys)
        else:
            raise Exception(
                f'Invalid mode: {mode}, please select from "train", "val", and "test".'
            )

    def preprocess(self, x, y):
        assert len(x.shape) == 3, "Incorrect data shape!"

        # var_idx = [7, 8, 11, 14, 15, -1] #[3, 6, 10, 14, 15] # needs adjusting for the daily averaged datasets [7, 8, 11, 14, 15]

        # x = x[0]
        # y = y[0]
        if self.x_noise:
            x = add_noise(
                x,
                [
                    0.0,
                    np.sqrt(0.0001),
                    np.sqrt(0.25),
                    np.sqrt(0.0025),
                    np.sqrt(0.0025),
                    0.0,
                ],
            )
        if self.y_noise:
            y = add_noise(
                y,
                [
                    0.0,
                    np.sqrt(0.0001),
                    np.sqrt(0.25),
                    np.sqrt(0.0025),
                    np.sqrt(0.0025),
                    0.0,
                ],
            )

        if self.transform:
            x = self.scaler.transform(x)
            y = self.scaler.transform(y)

        bc_mask = np.broadcast_to(self.mask[..., np.newaxis], x.shape)

        x[bc_mask] = 0
        y[bc_mask] = 0

        x_in = np.transpose(x, axes=[2, 0, 1])
        x_out = np.transpose(y, axes=[2, 0, 1])[:-1, ...]
        return (x_in, x_out)

    def __len__(self):
        return len(self.keys) * (
            self.time_steps_per_forward - 1
        )  # b/c n  time steps can create n-1 input-output pairs

    def __getitem__(self, index):
        # get the key idx
        key_idx = int(index / (self.time_steps_per_forward - 1))
        in_group_idx = index % (self.time_steps_per_forward - 1)
        data_x = self.data[self.keys[key_idx]][in_group_idx]
        data_y = self.data[self.keys[key_idx]][in_group_idx + 1]
        x, y = self.preprocess(data_x, data_y)
        assert not np.any(np.isnan(x)) and not np.any(
            np.isnan(y)
        ), "Data contains NaNs!!!"
        return (
            torch.from_numpy(x[1:, ...]).float(),
            torch.from_numpy(y[1:, ...]).float(),
        )


if __name__ == "__main__":
    # data = SOMAdata(
    #     "/Users/yixuan.sun/Documents/Projects/ImPACTS/deep_ensemble/datasets/GM-prog-var-surface-noise.hdf5",
    #     "train",
    # )

    data = h5py.File(
        "../../../deep_ensemble/datasets/GM-prog-var-surface.hdf5",
        "r",
    )

    with h5py.File(
        "../../../deep_ensemble/datasets/GM-prog-var-surface-noise.hdf5", "w"
    ) as f:
        for key in data.keys():
            print(key)
            sample_data = data[key][...]

            noise_data = add_noise(
                sample_data,
                [
                    0.0,
                    np.sqrt(0.0001),
                    np.sqrt(0.25),
                    np.sqrt(0.0025),
                    np.sqrt(0.0025),
                    0.0,
                ],
            )
            print(noise_data.shape)
            f.create_dataset(key, data=noise_data)

    # mask = np.abs(sample_data) > 1e16
    # print(sample_data.shape)
    # sample_data[mask] = np.nan
    # nosie_data[mask] = np.nan

    # import matplotlib.pyplot as plt
    #
    # fig, axs = plt.subplots(1, 3)
    # var_idx = 0
    # im = axs[0].imshow(sample_data[0, :, :, var_idx])
    # im2 = axs[1].imshow(nosie_data[0, :, :, var_idx])
    # im3 = axs[2].imshow(
    #     nosie_data[0, :, :, var_idx] - sample_data[0, :, :, var_idx]
    # )
    #
    # plt.colorbar(im)
    # plt.colorbar(im2)
    # plt.colorbar(im3)
    # plt.show()
    #
    # print(nosie_data.shape)
