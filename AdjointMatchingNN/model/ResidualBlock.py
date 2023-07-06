import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_hl) -> None:
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # create layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim)) # input layer
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())

        self.mapLayer = nn.Linear(in_dim, out_dim)
        for i in range(num_hl):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        if self.in_dim != self.out_dim:
            residual = self.mapLayer(x)
        else:
            residual = x
        for layer in self.layers:
            x = layer(x)
        out = residual + x
        return out

class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_dim, num_hl) -> None:
        super(ResidualBlock3D, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # create layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv3d(in_ch, hidden_dim, 3, padding='same')) # input layer
        self.layers.append(nn.BatchNorm3d(hidden_dim))
        self.layers.append(nn.ReLU())

        self.mapLayer = nn.Conv3d(in_ch, out_ch, 3)
        for i in range(num_hl):
            self.layers.append(nn.Conv3d(hidden_dim, hidden_dim, 3, padding='same'))
            self.layers.append(nn.BatchNorm3d(hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv3d(hidden_dim, out_ch, 3, padding='same'))
    def forward(self, x):
        if self.in_ch != self.out_ch:
            residual = self.mapLayer(x)
        else:
            residual = x
        for layer in self.layers:
            x = layer(x)
        out = residual + x
        return out




