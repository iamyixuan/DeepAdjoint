import os
import pickle
import torch
import numpy as np
import argparse

from deep_adjoint.model.ForwardSurrogate import OneStepSolve3D
from deep_adjoint.utils.data import SOMAdata
from deep_adjoint.train.trainer import Trainer, ddp_setup, mp, destroy_process_group, predict

def run(rank, world_size, args):
    ddp_setup(rank, world_size)


    # if torch.cuda.is_available():
    #     device = torch.device("cuda")  # Use CUDA device
    # else:
    #     device = torch.device("cpu")

    net = OneStepSolve3D(in_ch=17,
                        out_ch=16, 
                        hidden=args.hidden, 
                        num_res_block=args.num_res_block)
    trainer = Trainer(net=net, 
                      optimizer_name='Adam', 
                      loss_name='MSE',
                      gpu_id=rank)
    data_path = '/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/thedataset2.hdf5'
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
                  mask=args.mask)
        destroy_process_group()
    else:
        true, pred, gm = predict(net=net, test_data=test_set, gpu_id=rank,
                     checkpoint='./checkpoints/2023-07-26_SOMA-ResNet-mask/model_saved_ep_380')
        with open('/pscratch/sd/y/yixuans/2023-07-27-true_pred-masked.pkl', 'wb') as f:
            true_pred = {'true': true, 'pred': pred, 'gm': gm}
            pickle.dump(true_pred, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden', default=2, type=int)
    parser.add_argument('-num_res_block', default=2, type=int)
    parser.add_argument('-epochs', default=10, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-train', default='True', type=str)
    parser.add_argument('-mask', default=None, type=str)
    parser.add_argument('-model_name', type=str, default='test')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if args.train == 'True':
        mp.spawn(run, args=(world_size, args), nprocs=world_size)
    else:
        run(0, 1, args)
