from deep_adjoint.train.trainer import rvs_adjoint
import torch
from neuralop.models import TFNO3d
from deep_adjoint.utils.data import SOMAdata
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import h5py

MODEL_PATH = 'checkpoints/2023-09-05_GM-FNO-5state/model_saved_ep_2430'
model = TFNO3d(n_modes_height = 4, n_modes_width = 4, n_modes_depth = 4, in_channels = 6, out_channels = 5, hidden_channels = 16, projection_channels = 32, factorization = 'tucker', rank = 0.42)
model.load_state_dict(torch.load(MODEL_PATH))

data_path = '/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-GM.hdf5'
data_path_model = '/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/thedataset-GM_FD_0.001.hdf5'
test_set = SOMAdata(path=data_path, mode='test', gpu_id=0)
scaler = test_set.scaler
mask1 = test_set.mask1
mask2 = test_set.mask2
mask = np.logical_or(mask1, mask2)


test_data = DataLoader(test_set, batch_size=1)

def sample_locs(n_samples, mask, x_range=100, y_range=100):
    '''
    sample horizontal locations for adjoint calculation
    we only sample x, y locations as of now and use a fixed
    vertical depth for reporting the results
    '''
    sampled_locs = []
    while len(sampled_locs) < n_samples:
        x_sample = np.random.randint(low=0, high=x_range)
        y_sample = np.random.randint(low=0, high=y_range)
        if mask[x_sample][y_sample] == False: # check if the sampled location is within the domain of interest
            sampled_locs.append((x_sample, y_sample))
    return sampled_locs
           

def rvs_adjoint(model, inputs, v, locs):
    inputs.requires_grad = True 
    model.eval()
    out = model(inputs)
    out_loc = out[0, v, locs[0], locs[1], locs[2]] # t = 15
    out_loc.backward()
    J = inputs.grad
    J = torch.mean(J, (0, 2, 3, 4))[-1]
    return J.numpy()

def get_true_adj(path, locs):
    data = h5py.File(path, 'r')
    f_plus_x = data['forward_1'][...]
    f_minus_x = data['forward_2'][...]
    
    mask1 = f_plus_x > 1e16
    mask2 = f_plus_x < -1e16
    mask = np.logical_or(mask1, mask2)
    del_x = (f_plus_x[-1,...] - f_minus_x[-1,...]).mean()

    

    
    # assert (f_minus_x[...,-1] == f_plus_x[...,-1]).all(), 'they are different'
    f_plus_x[mask] = 0
    f_minus_x[mask] = 0
    # f_plus_x = scaler.transform(f_plus_x)
    # f_minus_x = scaler.transform(f_minus_x)

    var_idx = [3, 6, 10, 14, 15] 
    #var_idx_in = var_idx + [-1]
 
    f_plus_x = np.transpose(f_plus_x, axes=[4, 0, 1, 2, 3])[var_idx, ...]
    f_minus_x = np.transpose(f_minus_x, axes=[4, 0, 1, 2, 3])[var_idx, ...]


    print(del_x)
    df_dx = (f_plus_x[:-1, ...] - f_minus_x[:-1,...]) / del_x
    print(df_dx)

    return df_dx
    


if __name__=="__main__":
    i = 0
    adj = {}
    mask_new = mask[0, 0, :, :, 0]
    locs_sampled = sample_locs(5, mask_new)
    for i,l in enumerate(locs_sampled):
        adj[f'loc_{i}'] = l
        adj[f'adj_{i}'] = []
        for x, y in tqdm(test_data):
            adj_var = []
            for v in range(5):
                locs = [l[0], l[1], 30]
                out_adjoint = rvs_adjoint(model, x, v, locs)
                adj_var.append(out_adjoint)
            adj[f'adj_{i}'].append(np.array(adj_var))
    import pickle
    with open('./pred_results/adjRandLocs.pkl', 'wb') as f:
        pickle.dump(adj, f)

    # adj = np.array(adj)
    # np.save('adj_more_locs.npy', adj)
    # print(adj.shape)
    # true_adj = get_true_adj(data_path_model, [30, 50, 50])
    # true_adj = np.array(true_adj)
    # np.save('GM_FD_0.001_adj.npy', true_adj)

    # print(true_adj.shape)
    
