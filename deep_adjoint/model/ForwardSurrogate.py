import torch
import torch.nn as nn
from .ResidualBlock import ResidualBlock, ResidualBlock3D

class ResnetSurrogate(nn.Module):
    def __init__(self, time_steps, in_dim, out_dim, h_dim) -> None:
        super(ResnetSurrogate, self).__init__()
        self.layers = nn.ModuleList()
        for t in range(time_steps):
            if t == 0:
                self.layers.append(ResidualBlock(in_dim, out_dim, h_dim, 3))
            else:
                self.layers.append(ResidualBlock(out_dim, out_dim, h_dim, 3))

    def forward(self, x):
        self.sol = []
        for layer in self.layers:
            x = layer(x)
            # x.retain_grad()
            self.sol.append(x)
        return torch.stack(self.sol, dim=1) # variable length solution, depending on the timesteps.

class OneStepSolve3D(nn.Module):
    def __init__(self, in_ch, out_ch, hidden, num_res_block) -> None:
        super(OneStepSolve3D, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv3d(in_ch, hidden, 3, padding='same'))
        self.layers.append(nn.Tanh())
        for i in range(num_res_block):
            self.layers.append(ResidualBlock3D(hidden, hidden, 2, 2))
        self.layers.append(nn.Conv3d(hidden, out_ch, 3, padding='same'))
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class FNN(nn.Module):
    def __init__(self, in_dim, out_dim, layer_sizes) -> None:
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [in_dim] + layer_sizes + [out_dim]
        for k in range(len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        