import torch
from torch.utils.data import DataLoader
from deep_adjoint.model.ResidualBlock import ResidualBlock3D
from deep_adjoint.model.ForwardSurrogate import ResnetSurrogate, OneStepSolve3D
from deep_adjoint.utils.losses import Losses
from deep_adjoint.utils.data import SOMAdata
from deep_adjoint.train.trainer import Trainer

if __name__ == "__main__":
    # net = ResnetSurrogate(
    #     time_steps=200,
    #     in_dim=129,
    #     out_dim=128,
    #     h_dim=20,

    # )

    # data = MultiStepData()
    # x, (y, adj) = data[:10]
    # print(x.shape, y.shape)
    # print(adj.shape)
    # val_data = MultiStepData(path='./AdjointMatchingNN/Data/mixed_nu/val/')

    # trainer = Trainer(net=net,
    #                   optimizer_name='Adam',
    #                   loss_name='MSE')
    # trainer.train(train=data,
    #               val=val_data,
    #               epochs=100,
    #               batch_size=10,
    #               learning_rate=0.01)

    import numpy as np
    # data = np.zeros(((1, 15, 185, 309, 60)))
    # data = torch.from_numpy(data).float()

    # net = OneStepSolve3D(15, 15, 20, 3)
    # a = net(data)
    # print(a.shape)

    ds = SOMAdata(path='/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/thedataset.hdf5', mode='train', device=torch.device('cpu'))
    print(ds.__len__())
    dl = DataLoader(ds, batch_size=36)
    for x, y in dl:
        print(np.isnan(x.cpu().numpy()).any())
        print(np.isnan(y.cpu().numpy().any()))
        print(x.max())
        
