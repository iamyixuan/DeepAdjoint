import argparse
import os
import pickle

import torch
from torch.distributed import init_process_group

from deep_adjoint.model.VAE import VAE
from deep_adjoint.train.trainer import (TrainerSOMA, destroy_process_group,
                                        pred_rollout)
from deep_adjoint.utils.data import SimpleDataset


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
    SCRATCH = "/pscratch/sd/y/yixuans/"
    train_set = SimpleDataset(file_path=SCRATCH + "temp_vae_train.h5")
    val_set = SimpleDataset(file_path=SCRATCH + "temp_vae_val.h5")
    test_set = SimpleDataset(file_path=SCRATCH + "temp_vae_test.h5")

    # ================= Load model =================

    net = VAE(
        input_ch=1,
        hidden_ch=32,
        latent_dim=args.latent_dim,
        scaler=None,
        train_data_stats=None,
    )

    trainer = TrainerSOMA(
        net=net,
        optimizer_name="Adam",
        loss_name="VAE",
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
            train_vae=True,
        )

        destroy_process_group()

        _, pred, x = trainer.predict(test_data=test_set)
        save_path = f"/pscratch/sd/y/yixuans/{trainer.exp_path}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(
            f"{save_path}/test-predictions.pkl",
            "wb",
        ) as f:
            true_pred = {"pred": pred, "x": x}
            pickle.dump(true_pred, f)

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
        raise ValueError("Invalid argument for train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", default=1000, type=int)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-lr", default=1e-5, type=float)
    parser.add_argument("-train", default="True", type=str)
    parser.add_argument("-model", type=str, default="4D")
    parser.add_argument("-load_model", action="store_true")
    parser.add_argument("-model_path", type=str)
    parser.add_argument("-save_freq", type=int, default=100)
    parser.add_argument("-data", type=str, default="GM_D_AVG")

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
