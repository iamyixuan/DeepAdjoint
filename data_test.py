from deep_adjoint.utils.data import SOMAdata
from torch.utils.data import DataLoader
data_path = '/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-GM-dayAvg-2.hdf5'
test_set = SOMAdata(path=data_path, mode='test', gpu_id=0)
loader = DataLoader(test_set, batch_size=1)
for x, y in loader:
    print(x.shape, y.shape)
    print(x[0,-1, 0, 50, 50])
    
