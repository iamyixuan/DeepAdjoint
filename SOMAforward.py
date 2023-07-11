import os
import torch
import numpy as np
import argparse

from deep_adjoint.model.ForwardSurrogate import OneStepSolve3D
from deep_adjoint.utils.data import SOMAdata
from deep_adjoint.train.trainer import Trainer

def run(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA device
    else:
        device = torch.device("cpu")
    net = OneStepSolve3D(in_ch=15,
                        out_ch=14, 
                        hidden=args.hidden, 
                        num_res_block=args.num_res_block)
    trainer = Trainer(net=net, 
                      optimizer_name='Adam', 
                      loss_name='MSE')
    data_path = '/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/thedataset.hdf5'
    train_set = SOMAdata(path=data_path, mode='train', device=device) 
    val_set = SOMAdata(path=data_path, mode='val', device=device) 
    test_set = SOMAdata(path=data_path, mode='test', device=device) 

    trainer.train(train=train_set,
                  val=val_set,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden', default=2, type=int)
    parser.add_argument('-num_res_block', default=2, type=int)
    parser.add_argument('-epochs', default=10, type=int)
    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    args = parser.parse_args()


    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA device
    else:
        device = torch.device("cpu")
    run(args)
