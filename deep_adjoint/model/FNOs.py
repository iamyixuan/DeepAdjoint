import torch.nn as nn
from neuralop.models import TFNO3d

from ..utils.scaler import ChannelStandardScaler


class FNO3d(nn.Module):
    def __init__(
        self,
        n_modes_height=4,
        n_modes_width=4,
        n_modes_depth=4,
        in_channels=6,
        out_channels=5,
        hidden_channels=16,
        projection_channels=32,
        **kwargs,
    ):
        super(FNO3d, self).__init__()
        self.fno = TFNO3d(
            n_modes_height=n_modes_height,
            n_modes_width=n_modes_width,
            n_modes_depth=n_modes_depth,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            projection_channels=projection_channels,
        )

        if kwargs["scaler"] is not None and kwargs["train_data"] is not None:
            train_data_x, train_data_y = kwargs["train_data"]
            self.scaler_x = ChannelStandardScaler(
                train_data_x, mask=kwargs["mask"]
            )
            self.scaler_y = ChannelStandardScaler(
                train_data_y, mask=kwargs["mask"]
            )

    def forward(self, x):
        if self.scaler_x:
            x = self.scaler_x.transform(x)
            x = self.fno(x)
            x = self.scaler_y.inverse_transform(x)
        else:
            x = self.fno(x)
        return x
