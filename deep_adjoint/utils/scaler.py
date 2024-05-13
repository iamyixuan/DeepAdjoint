import numpy as np
import torch


class ChannelStandardScaler:
    def __init__(self, data, **kwargs) -> None:
        self.init(data, **kwargs)

    def init(self, data, dim_to_reduce=(0, 2, 3, 4), **kwargs):
        if kwargs.get("mask") is not None:
            mask = kwargs.get("mask")
            bc_mask = np.broadcast_to(
                mask[np.newaxis, ..., np.newaxis], data.shape
            )
            data[bc_mask] = float('nan')  # setting values outside the domain to 0

        m_ = torch.nanmean(data, dim=dim_to_reduce, keepdims=True)
        std_ = torch.nanstd(data, dim=dim_to_reduce, keepdims=True)

        std_[std_ == 0] = 1

        self.m_ = m_
        self.std_ = std_

    def transform(self, x):
        x = (x - self.m_) / self.std_
        x[torch.isnan(x)] = 0
        return x

    def inverse_transform(self, x):
        x = x * self.std_ + self.m_
        x[torch.isnan(x)] = 0
        return x


class MinMaxScaler:
    def __init__(self, data, min_=0, max_=1) -> None:
        # data = data.astype(np.float64)
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
        self.data_min = data_min.reshape(1, 1, 1, data_min.shape[0])
        self.data_max = data_max.reshape(1, 1, 1, data_max.shape[0])
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
