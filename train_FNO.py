import argparse
import os
import pickle

import torch
from torch.distributed import init_process_group

from deep_adjoint.model.FNOs import FNO3d, FNO4d
from deep_adjoint.train.trainer import (TrainerSOMA, destroy_process_group,
                                        pred_rollout)
from deep_adjoint.utils.data import SOMAdata


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

    # ================= Load data =================

    if args.data == "GM":
        data_path = "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset3.hdf5"
    elif args.data == "REDI":
        data_path = (
            "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-redi-2.hdf5"
        )
    elif args.data == "CVMIX":
        data_path = (
            "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-cvmix-2.hdf5"
        )
    elif args.data == "BTTMDRAG":
        data_path = "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-impliciBottomDrag.hdf5"
    elif args.data == "GM_D_AVG":
        data_path = (
            "/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-GM-dayAvg-2.hdf5"
        )
    else:
        raise TypeError("Dataset not recognized!")

    train_set = SOMAdata(path=data_path, horizon=5, var_idx=[8], mode="train")
    val_set = SOMAdata(path=data_path, horizon=5, var_idx=[8], mode="val")
    test_set = SOMAdata(path=data_path, horizon=5, var_idx=[8], mode="test")

    # ================= Load model =================

    if args.model == "3D":
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
    elif args.model == "4D":
        net = FNO4d(
            n_modes_height=4,
            n_modes_width=4,
            n_modes_depth=4,
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            projection_channels=32,
            scaler=True,
            train_data_stats=(train_set.mean, train_set.std),
            gpu_id=args.gpu,
            mask=train_set.mask,
        )

    trainer = TrainerSOMA(
        net=net,
        optimizer_name="Adam",
        loss_name="MSE",
        gpu_id=args.gpu,
        model=args.model,
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
            load_model=False,
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
    parser.add_argument("-epochs", default=1000, type=int)
    parser.add_argument("-batch_size", default=8, type=int)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-train", default="True", type=str)
    parser.add_argument("-model", type=str, default="4D")
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
