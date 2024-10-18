"""
This script is to train a model based on specified configuration and
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from data import SimpleDataset, SOMAdata
from metrics import MAELoss
from model import Trainer
from modulus.models.fno import FNO
from torch.utils.data import DataLoader




def run(config, args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    net = FNO(
        in_channels=5,
        out_channels=4,
        decoder_layers=config["num_projs"],
        decoder_layer_size=config["proj_size"],
        decoder_activation_fn=config["proj_act"],
        dimension=2,
        latent_channels=config["latent_ch"],
        num_fno_layers=config["num_FNO"],
        num_fno_modes=int(config["num_modes"]),
        padding=config["padding"],
        padding_type=config["padding_type"],
        activation_fn=config["lift_act"],
        coord_features=config["coord_feat"],
    )

    filename = "GM-prog-var-surface.hdf5"
    trainset = SOMAdata(
        f"/pscratch/sd/y/yixuans/datatset/de_dataset/{filename}",
        "train",
        y_noise=True,
        transform=True,
    )
    valset = SOMAdata(
        f"/pscratch/sd/y/yixuans/datatset/de_dataset/{filename}",
        "val",
        transform=True,
    )
    testSet = SOMAdata(
        f"/pscratch/sd/y/yixuans/datatset/de_dataset/{filename}",
        "test",
        transform=True,
    )

    # TrainLossFn = MSE_ACCLoss(alpha=config['alpha'])
    TrainLossFn = MAELoss()
    ValLossFn = MAELoss()

    trainer = Trainer(model=net, TrainLossFn=TrainLossFn, ValLossFn=ValLossFn)

    trainLoader = DataLoader(
        trainset, batch_size=int(config["batch_size"]), shuffle=True
    )
    valLoader = DataLoader(valset, batch_size=valset.__len__())
    testLoader = DataLoader(testSet, batch_size=1)

    # train if the env variable is set
    if os.environ.get("TRAIN") == "True":
        log, model = trainer.train(
            trainLoader=trainLoader,
            valLoader=valLoader,
            epochs=500,
            optimizer=config["optimizer"],
            learningRate=1e-4,  # config["lr"],
            weight_decay=config["weight_decay"],
        )


        # save the model
        torch.save(
            trainer.model.state_dict(),
            f"./experiments/{args.ensemble_id}_bestStateDict_mae",
        )
        true, pred = trainer.predict(testLoader)
        np.savez(
            f"./experiments/{args.ensemble_id}_Pred_mae.npz",
            true=true,
            pred=pred,
        )

        # trainLoss = log.logger["TrainLoss"]
        # valLoss = log.logger["ValLoss"]

        with open(
            f"./experiments/{args.ensemble_id}_train_curve_mae.pkl", "wb"
        ) as f:
            pickle.dump(log.logger, f)

    elif os.environ.get("ROLLOUT") == "True":
        # load the model
        print("Pefroming Rollout...")
        trainer.model.load_state_dict(
            torch.load(f"./experiments/{args.ensemble_id}_bestStateDict_mae")
        )
        rolloutPred = trainer.rollout(testLoader)
        with open(
            f"./experiments/{args.ensemble_id}_Rollout_mae.pkl", "wb"
        ) as f:
            pickle.dump(rolloutPred, f)
    return


def getBestConfig(df, ensemble_id):
    # load the result.csv datafraem
    # remove 'F'
    df = df[df["objective_0"] != "F"]
    # df = df[df['objective_2']!='F']
    # take out the pareto front
    # df = df[df['pareto_efficient']==True]

    # convert to float and take the negative
    df["objective_0"] = df["objective_0"].astype(float)
    df["objective_1"] = df["objective_1"].astype(float)
    # Pick the best objectives

    max_row_index = (
        (df["objective_0"] + df["objective_1"])
        .sort_values(ascending=False)
        .index[ensemble_id]
    )
    df = df.rename(columns=lambda x: x.replace("p:", ""))
    print("config acc is", df.loc[max_row_index]["objective_1"])
    return df.loc[max_row_index]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--df_path", type=str, default="./results/results.csv")
    parser.add_argument("--ensemble_id", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(
        args.df_path,
    )
    config = dict(getBestConfig(df, args.ensemble_id))
    run(config, args)
