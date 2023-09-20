import os
import pickle
import builtins
import torch
import numpy as np
import argparse

from deep_adjoint.model.ForwardSurrogate import OneStepSolve3D, FNN
from torch.distributed import init_process_group
from neuralop.models import TFNO3d
from neuralop.models import FNO3d
from pytorch3dunet.unet3d.model import UNet3D
from deep_adjoint.utils.data import SOMAdata, SOMA_PCA_Data
from deep_adjoint.train.trainer import Trainer, ddp_setup, mp, destroy_process_group, predict, pred_rollout

def run(args):
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.gpu = 0 
        

    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.net_type == 'ResNet':
        net = OneStepSolve3D(in_ch=6,
                            out_ch=5, 
                            hidden=args.hidden, 
                            num_res_block=args.num_res_block)
    elif args.net_type == 'UNet':
        net = UNet3D(in_channels=6, out_channels=5)
    elif args.net_type == 'FNN':
        net = FNN(in_dim=50*5, out_dim=50*5, layer_sizes=[500]*20)
    elif args.net_type == 'FNO':
        net = TFNO3d(n_modes_height = 4, n_modes_width = 4, n_modes_depth = 4, in_channels = 6, out_channels = 5, hidden_channels = 16, projection_channels = 32, factorization = 'tucker', rank = 0.42)
    else:
        raise TypeError('Specify a network type!')
    
    trainer = Trainer(net=net, 
                      optimizer_name='Adam', 
                      loss_name='MSE',
                      gpu_id=args.gpu)

    if args.net_type == 'FNN':
        print('Using PCA data...')
        data_path = '/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/PCA_data.h5'
        train_set = SOMA_PCA_Data(path=data_path, mode='train')
        val_set = SOMA_PCA_Data(path=data_path, mode='val')
        test_set = SOMA_PCA_Data(path=data_path, mode='test')
    else:
        data_path = '/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/thedataset3.hdf5'
        train_set = SOMAdata(path=data_path, mode='train', gpu_id=args.gpu, train_noise=True) 
        val_set = SOMAdata(path=data_path, mode='val', gpu_id=args.gpu, train_noise=True) 
        test_set = SOMAdata(path=data_path, mode='test', gpu_id=args.gpu) 
    
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
        true, pred, gm = predict(net=net, test_data=test_set, gpu_id=0,
                     checkpoint=args.model_path)
        with open('/pscratch/sd/y/yixuans/2023-9-20-var-pred-FNO-trNoise.pkl', 'wb') as f:
            true_pred = {'true': true, 'pred': pred, 'gm': gm}
            pickle.dump(true_pred, f)
    else:
        true, pred = pred_rollout(net=net, test_data=test_set, gpu_id=args.gpu, checkpoint=args.model_path)
        with open('/pscratch/sd/y/yixuans/FNO-5var-rollout-trNoise.pkl', 'wb') as f:
            rollout = {'true': true, 'pred': pred}
            pickle.dump(rollout, f)
        print(true.shape, pred.shape)


if __name__ == "__main__":
    import hostlist
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden', default=2, type=int)
    parser.add_argument('-num_res_block', default=2, type=int)
    parser.add_argument('-epochs', default=1000, type=int)
    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-train', default='True', type=str)
    parser.add_argument('-mask', default=None, type=str)
    parser.add_argument('-model_name', type=str, default='test')
    parser.add_argument('-net_type', type=str, default='UNet')
    parser.add_argument('-load_model', action='store_true')
    parser.add_argument('-model_path', type=str)
    parser.add_argument('-save_freq', type=int, default=10)

    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local-rank', default=-1, type=int, 
                        help='local rank for distributed training')

    args = parser.parse_args()

    if "WORLD_SIZE" in os.environ:
        WORLD_SIZE = int(os.environ['WORLD_SIZE']) 
        ngpus_per_node = torch.cuda.device_count()
        if 'SLURM_PROCID' in os.environ:
            RANK = int(os.environ['SLURM_PROCID'])
            gpu = RANK % ngpus_per_node
        else:
            RANK = int(os.environ['LOCAL_RANK'])
            gpu = int(os.environ['LOCAL_RANK'])
  
    if args.train == 'True':
        # mp.spawn(run, args=(world_size, args), nprocs=world_size)
        run(args)
    else:
        os.environ['TRAIN'] = '0'
        run(args)
