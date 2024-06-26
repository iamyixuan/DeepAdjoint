import argparse
import os
import pickle

import torch
from torch.distributed import init_process_group

from deep_adjoint.model.FNOs import FNO3d
from deep_adjoint.model.ForwardSurrogate import FNN, OneStepSolve3D
from deep_adjoint.train.trainer import TrainerSOMA, destroy_process_group, pred_rollout
from deep_adjoint.utils.data import SOMA_PCA_Data, SOMAdata
from pytorch3dunet.unet3d.model import UNet3D


def run(args):
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    print("WORLD_SIZE:", args.world_size)
    args.distributed = args.world_size > 1
    # ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif "SLURM_PROCID" in os.environ:  # for slurm scheduler
            args.rank = int(os.environ["SLURM_PROCID"])
            args.gpu = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        print("Initializing process group...")
        init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        print("Done.")
    else:
        args.gpu = 0

    # if args.rank!=0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass
    if args.net_type == "FNN":
        print("Using PCA data...")
        data_path = "/pscratch/sd/y/yixuans/datatset/SOMA/varyGM/PCA_data.h5"
        train_set = SOMA_PCA_Data(path=data_path, mode="train")
        val_set = SOMA_PCA_Data(path=data_path, mode="val")
        test_set = SOMA_PCA_Data(path=data_path, mode="test")
    else:
        if args.data == "GM":
            data_path = "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset3.hdf5"
        elif args.data == "REDI":
            data_path = "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-redi-2.hdf5"
        elif args.data == "CVMIX":
            data_path = "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-cvmix-2.hdf5"
        elif args.data == "BTTMDRAG":
            data_path = (
                "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-impliciBottomDrag.hdf5"
            )
        elif args.data == "GM_D_AVG":
            data_path = (
                "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-GM-dayAvg-2.hdf5"
            )
        else:
            raise TypeError("Dataset not recognized!")

        train_set = SOMAdata(path=data_path, mode="train")
        val_set = SOMAdata(path=data_path, mode="val")
        test_set = SOMAdata(path=data_path, mode="test")

    if args.net_type == "ResNet":
        net = OneStepSolve3D(
            in_ch=6,
            out_ch=5,
            hidden=args.hidden,
            num_res_block=args.num_res_block,
        )
    elif args.net_type == "UNet":
        net = UNet3D(in_channels=6, out_channels=5)
    elif args.net_type == "FNN":
        net = FNN(in_dim=50 * 5, out_dim=50 * 5, layer_sizes=[500] * 20)
    elif args.net_type == "FNO":
        net = FNO3d(
            n_modes_height=4,
            n_modes_width=4,
            n_modes_depth=4,
            in_channels=6,
            out_channels=5,
            hidden_channels=16,
            projection_channels=32,
            scaler=True,
            train_data_stats=(train_set.mean, train_set.std),
            gpu_id=args.gpu,
            mask=train_set.mask,
        )
    else:
        raise TypeError("Specify a network type!")

    trainer = TrainerSOMA(
        net=net,
        optimizer_name="Adam",
        loss_name="MSE",
        gpu_id=args.gpu,
        model_type=args.net_type,
        data_name=args.data,
    )

    if args.train == "True":
        trainer.train(
            train=train_set,
            val=val_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            mask=None,
            save_freq=args.save_freq,
            model_path=args.model_path,
            load_model=True,
        )

        destroy_process_group()

    elif args.train == "False":
        true, pred, param = trainer.predict(
           test_data=test_set, checkpoint=args.model_path
        )
        save_path = f"/pscratch/sd/y/yixuans/{trainer.exp_path}"
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(
            f"{save_path}/test-predictions.pkl",
            "wb",
        ) as f:
            true_pred = {"true": true, "pred": pred, "param": param}
            pickle.dump(true_pred, f)
    else:
        true, pred = pred_rollout(
            net=net,
            test_data=test_set,
            gpu_id=args.gpu,
            checkpoint=args.model_path,
        )
        with open(
            f"{trainer.exp_path}/test-rollout.pkl",
            "wb",
        ) as f:
            rollout = {"true": true, "pred": pred}
            pickle.dump(rollout, f)
        print(true.shape, pred.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-hidden", default=2, type=int)
    parser.add_argument("-num_res_block", default=2, type=int)
    parser.add_argument("-epochs", default=1000, type=int)
    parser.add_argument("-batch_size", default=8, type=int)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-train", default="True", type=str)
    parser.add_argument("-mask", default="True", type=str)
    parser.add_argument("-model_name", type=str, default="test")
    parser.add_argument("-net_type", type=str, default="UNet")
    parser.add_argument("-load_model", action="store_true")
    parser.add_argument("-model_path", type=str)
    parser.add_argument("-save_freq", type=int, default=100)
    parser.add_argument("-data", type=str, default="GM")

    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank",
        default=-1,
        type=int,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--local-rank",
        default=-1,
        type=int,
        help="local rank for distributed training",
    )

    args = parser.parse_args()

    if "WORLD_SIZE" in os.environ:
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        ngpus_per_node = torch.cuda.device_count()
        if "SLURM_PROCID" in os.environ:
            RANK = int(os.environ["SLURM_PROCID"])
            gpu = RANK % ngpus_per_node
        else:
            RANK = int(os.environ["LOCAL_RANK"])
            gpu = int(os.environ["LOCAL_RANK"])

    if args.train == "True":
        # mp.spawn(run, args=(world_size, args), nprocs=world_size)
        run(args)
    else:
        os.environ["TRAIN"] = "0"
        run(args)
