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
from metrics import QuantileLoss
from model import Trainer
from modulus.models.fno import FNO
from torch.utils.data import DataLoader


class FNO_QR(torch.nn.Module):
    def __init__(
        self,
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
        super(FNO_QR, self).__init__()
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
        return x


def run(config, args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    net = FNO_QR(
        in_channels=5,
        out_channels=4 * 3,
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
    TrainLossFn = QuantileLoss()
    ValLossFn = QuantileLoss()

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
            epochs=1000,
            optimizer=config["optimizer"],
            learningRate=1e-4,  # config["lr"],
            weight_decay=config["weight_decay"],
        )

        # with open(
        #     f"./experiments/ensemble{args.ensemble_id}_log_quantile.pkl", "wb"
        # ) as f:
        #     pickle.dump(log.logger, f)

        # save the model
        torch.save(
            model.state_dict(),
            f"./experiments/{args.ensemble_id}_bestStateDict_quantile",
        )
        true, pred = trainer.predict(testLoader)
        np.savez(
            f"./experiments/{args.ensemble_id}_Pred_quantile.npz",
            true=true,
            pred=pred,
        )

        trainLoss = log.logger["TrainLoss"]
        valLoss = log.logger["ValLoss"]

        with open(
            f"./experiments/{args.ensemble_id}_train_curve_quantile.pkl", "wb"
        ) as f:
            pickle.dump({"train": trainLoss, "val": valLoss}, f)

    elif os.environ.get("ROLLOUT") == "True":
        # load the model
        print("Pefroming Rollout...")
        trainer.model.load_state_dict(
            torch.load(
                f"./experiments/{args.ensemble_id}_bestStateDict_quantile"
            )
        )
        rolloutPred = trainer.rollout(testLoader)
        with open(
            f"./experiments/{args.ensemble_id}_Rollout_qunatile.pkl", "wb"
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
