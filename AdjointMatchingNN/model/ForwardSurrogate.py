import torch
import torch.nn as nn
from .ResidualBlock import ResidualBlock

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