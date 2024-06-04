"""
This script is to train a model based on specified configuration and
"""

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from modulus.models.fno import FNO
from torch.utils.data import DataLoader

from data import SOMAdata
from metrics import NLLLoss
from model import Trainer


def run(config, args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    net = FNO(
        in_channels=6,
        out_channels=5 * 2,
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

    trainset = SOMAdata(
        args.data_path,
        "train",
        transform=True,
    )
    valset = SOMAdata(
        args.data_path,
        "val",
        transform=True,
    )
    testSet = SOMAdata(
        args.data_path,
        "test",
        transform=True,
    )

    # TrainLossFn = MSE_ACCLoss(alpha=config['alpha'])
    TrainLossFn = NLLLoss()
    ValLossFn = NLLLoss()

    trainer = Trainer(model=net, TrainLossFn=TrainLossFn, ValLossFn=ValLossFn)

    trainLoader = DataLoader(
        trainset, batch_size=int(config["batch_size"]), shuffle=True
    )
    valLoader = DataLoader(valset, batch_size=valset.__len__())
    testLoader = DataLoader(testSet, batch_size=testSet.__len__())

    log = trainer.train(
        trainLoader=trainLoader,
        valLoader=valLoader,
        epochs=100,
        optimizer=config["optimizer"],
        learningRate=config["lr"],
        weight_decay=config["weight_decay"],
    )
    time_now = datetime.now().strftime("%Y%m%d_%H-%M")

    with open(f"{time_now}_ensemble{args.ensemble_id}_log.pkl", "wb") as f:
        pickle.dump(log.logger, f)

    pred, true = trainer.predict(testLoader)
    np.savez(f"{time_now}_{args.ensemble_id}Pred.npz", true=true, pred=pred)

    trainLoss = log.logger["TrainLoss"]
    valLoss = log.logger["ValLoss"]

    with open(f"{time_now}_{args.ensemble_id}_train_curve.pkl", "wb") as f:
        pickle.dump({"train": trainLoss, "val": valLoss}, f)

    # rolloutPred = trainer.rollout(rolloutLoader)
    # with open(f"{time_now}_{config_name}Rollout.pkl", "wb") as f:
    #     pickle.dump(rolloutPred, f)
    return


def getBestConfig(df, ensemble_id):
    # load the result.csv datafraem
    # remove 'F'
    df = df[df["objective_0"] != "F"]
    # df = df[df['objective_2']!='F']
    # take out the pareto front
    # df = df[df['pareto_efficient']==True]

    # convert to float and take the negative
    # df["objective"] = -df["objective"].astype(float)
    # Pick the best objectives

    min_row_index = (
        (df["objective_0"] + df["objective_1"])
        .sort_values()
        .index[ensemble_id]
    )
    df = df.rename(columns=lambda x: x.replace("p:", ""))
    return df.loc[min_row_index]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--df_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--ensemble_id", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(
        args.df_path,
    )
    config = dict(getBestConfig(df, args.ensemble_id))
    run(config, args)
