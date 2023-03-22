import numpy as np
import tensorflow as tf
from PINN import *
from adjoint import *


N = 128
DX = 2/N
STEPS = 32
DT = 1/STEPS
NU = 0.01/(N*np.pi)

PINN_net = PINN(num_layers=10, hidden_dim=20, output_dim=1, act_fn='tanh')
PINN_DP_net = PINN(num_layers=10, hidden_dim=20, output_dim=1, act_fn='tanh')

PINN_net.load_weights('./saved_model/PINN_weights_epoch_100')
PINN_DP_net.load_weights('./saved_model/PINN_DP_weights_epoch_100')