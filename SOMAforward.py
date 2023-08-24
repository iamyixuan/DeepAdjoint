import os
import pickle
import torch
import numpy as np
import argparse

from deep_adjoint.model.ForwardSurrogate import OneStepSolve3D, FNN
from pytorch3dunet.unet3d.model import UNet3D
from deep_adjoint.utils.data import SOMAdata, SOMA_PCA_Data
from deep_adjoint.train.trainer import Trainer, ddp_setup, mp, destroy_process_group, predict, pred_rollout

def run(rank, world_size, args):
    ddp_setup(rank, world_size)


    # if torch.cuda.is_available():
    #     device = torch.device("cuda")  # Use CUDA device
    # else:
    #     device = torch.device("cpu")

    if args.net_type == 'ResNet':
        net = OneStepSolve3D(in_ch=17,
                            out_ch=5, 
                            hidden=args.hidden, 
                            num_res_block=args.num_res_block)
    elif args.net_type == 'UNet':
        net = UNet3D(in_channels=17, out_channels=5)
    elif args.net_type == 'FNN':
        net = FNN(in_dim=50*5, out_dim=50*5, layer_sizes=[500]*20)
    else:
        raise TypeError('Specify a network type!')
    
    trainer = Trainer(net=net, 
                      optimizer_name='Adam', 
                      loss_name='MSE',
                      gpu_id=rank)

    if args.net_type == 'FNN':
        print('Using PCA data...')
        data_path = '/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/PCA_data.h5'
        train_set = SOMA_PCA_Data(path=data_path, mode='train')
        val_set = SOMA_PCA_Data(path=data_path, mode='val')
        test_set = SOMA_PCA_Data(path=data_path, mode='test')
    else:
        data_path = '/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/thedataset3.hdf5'
        train_set = SOMAdata(path=data_path, mode='train', gpu_id=rank) 
        val_set = SOMAdata(path=data_path, mode='val', gpu_id=rank) 
        test_set = SOMAdata(path=data_path, mode='test', gpu_id=rank) 
    
    if args.train == "True":
        trainer.train(train=train_set,
                  val=val_set,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  model_name=args.model_name,
                  mask=args.mask,
                  save_freq=args.save_freq,
                  load_model=args.load_model,
                  model_path=args.model_path)
                
        destroy_process_group()
    elif args.train == "False":
        true, pred, gm = predict(net=net, test_data=test_set, gpu_id=rank,
                     checkpoint=args.model_path)
        with open('/pscratch/sd/y/yixuans/2023-8-16-ResNet-5var-predictions.pkl', 'wb') as f:
            true_pred = {'true': true, 'pred': pred, 'gm': gm}
            pickle.dump(true_pred, f)
    else:
        true, pred = pred_rollout(net=net, test_data=test_set, gpu_id=rank, checkpoint=args.model_path)
        print(true.shape, pred.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden', default=2, type=int)
    parser.add_argument('-num_res_block', default=2, type=int)
    parser.add_argument('-epochs', default=1000, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-train', default='True', type=str)
    parser.add_argument('-mask', default=None, type=str)
    parser.add_argument('-model_name', type=str, default='test')
    parser.add_argument('-net_type', type=str, default='ResNet')
    parser.add_argument('-load_model', action='store_true')
    parser.add_argument('-model_path', type=str)
    parser.add_argument('-save_freq', type=int, default=10)

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if args.train == 'True':
        mp.spawn(run, args=(world_size, args), nprocs=world_size)
    else:
        run(0, 1, args)
