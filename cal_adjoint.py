from deep_adjoint.train.trainer import rvs_adjoint
import torch
from neuralop.models import TFNO3d
from deep_adjoint.utils.data import SOMAdata
from torch.utils.data import DataLoader
from tqdm import tqdm

MODEL_PATH = 'checkpoints/2023-09-05_GM-FNO-5state/model_saved_ep_2430'
model = TFNO3d(n_modes_height = 4, n_modes_width = 4, n_modes_depth = 4, in_channels = 6, out_channels = 5, hidden_channels = 16, projection_channels = 32, factorization = 'tucker', rank = 0.42)
model.load_state_dict(torch.load(MODEL_PATH))

data_path = '/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/thedataset3.hdf5'
test_set = SOMAdata(path=data_path, mode='test', gpu_id=0)

test_data = DataLoader(test_set, batch_size=1)

def rvs_adjoint(model, inputs, locs):
    inputs.requires_grad = True
    # inputs.grad.zero_
    model.eval()
    out = model(inputs)
    out_loc = out[0, 0, 50, 50, 30]
    out_loc.backward()
    J = inputs.grad
    J = torch.mean(J, (0, 2, 3, 4))
    print(J)
    return J
 

nn_adjoint = torch.ones(size=[5])
i = 0
for x, y in tqdm(test_data):
    out_adjoint = rvs_adjoint(model, x, [0, 2, 3])
    nn_adjoint[i] = out_adjoint
    i += 1
    break

    
