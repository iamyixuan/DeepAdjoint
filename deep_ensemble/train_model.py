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

from data import SimpleDataset, SOMAdata
from metrics import NLLLoss
from model import Trainer

class FNO_MU_STD(torch.nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            decoder_layers,
            decoder_layer_size,
            decoder_activation_fn,
            dimension,
            latent_channels,
            num_fno_layers,
            num_fno_modes,
            padding,
            padding_type,
            activation_fn,
            coord_features,
        ):
        super(FNO_MU_STD, self).__init__()
        self.FNO = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            decoder_activation_fn=decoder_activation_fn,
            dimension=dimension,
            latent_channels=latent_channels,
            num_fno_layers=num_fno_layers,
            num_fno_modes=num_fno_modes,
            padding=padding,
            padding_type=padding_type,
            activation_fn=activation_fn,
            coord_features=coord_features,
        )
        self.std_act = torch.nn.Softplus()
    def forward(self, x):
        x = self.FNO(x)
        ch = x.shape[1] // 2
        mu = x[:, :ch, ...]
        std = self.std_act(x[:, ch:, ...])
        out = torch.cat([mu, std], dim=1)
        return out

def run(config, args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    net = FNO_MU_STD(
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

    # trainset = SimpleDataset(
    #     '/pscratch/sd/y/yixuans/datatset/de_dataset/soma_deep_ensemble_train.h5'
    #     )
    # valset = SimpleDataset(
    #     '/pscratch/sd/y/yixuans/datatset/de_dataset/soma_deep_ensemble_val.h5'
    # )
    # testSet = SimpleDataset(
    #     '/pscratch/sd/y/yixuans/datatset/de_dataset/soma_deep_ensemble_test.h5'
    # )

    trainset = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "train",
        transform=True,
    )
    valset = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "val",
        transform=True,
    )
    testSet = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
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
        epochs=500,
        optimizer=config["optimizer"],
        learningRate=1e-4, #config["lr"],
        weight_decay=config["weight_decay"],
    )

    with open(f"./experiments/ensemble{args.ensemble_id}_log2.pkl", "wb") as f:
        pickle.dump(log.logger, f)

    pred, true = trainer.predict(testLoader)
    np.savez(f"./experiments/{args.ensemble_id}_Pred2.npz", true=true, pred=pred)

    trainLoss = log.logger["TrainLoss"]
    valLoss = log.logger["ValLoss"]

    with open(f"./experiments/{args.ensemble_id}_train_curve2.pkl", "wb") as f:
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
    df["objective_0"] = df["objective_0"].astype(float)
    df["objective_1"] = df["objective_1"].astype(float)
    # Pick the best objectives

    max_row_index = (
        (df["objective_0"] + df["objective_1"])
        .sort_values(ascending=False)
        .index[ensemble_id]
    )
    df = df.rename(columns=lambda x: x.replace("p:", ""))
    print("config acc is", df.loc[max_row_index]['objective_1'])
    return df.loc[max_row_index]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--df_path", type=str, default='./results/results.csv')
    parser.add_argument("--ensemble_id", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(
        args.df_path,
    )
    config = dict(getBestConfig(df, args.ensemble_id))
    run(config, args)
