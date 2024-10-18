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
from metrics import NLLLoss
from model import Trainer
from modulus.models.fno import FNO
from torch.utils.data import DataLoader


class FNO_MU_STD(torch.nn.Module):
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
        beta=5.0,
        threshold=20.0,
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
        self.var_act = torch.nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x):
        x = self.FNO(x)
        ch = x.shape[1] // 2
        mu = x[:, :ch, ...]
        var = self.var_act(x[:, ch:, ...])
        out = torch.cat([mu, var], dim=1)
        return out


def run(config, args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    net = FNO_MU_STD(
        in_channels=5,
        out_channels=4 * 2,
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
        y_noise=True,
        transform=True,
    )
    testSet = SOMAdata(
        f"/pscratch/sd/y/yixuans/datatset/de_dataset/{filename}",
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
            f"./experiments/{args.ensemble_id}_bestStateDict_nll",
        )
        true, pred = trainer.predict(testLoader)
        np.savez(
            f"./experiments/{args.ensemble_id}_Pred_nll.npz",
            true=true,
            pred=pred,
        )

        # trainLoss = log.logger["TrainLoss"]
        # valLoss = log.logger["ValLoss"]

        with open(
            f"./experiments/{args.ensemble_id}_train_curve_nll.pkl", "wb"
        ) as f:
            pickle.dump(log.logger, f)

    elif os.environ.get("ROLLOUT") == "True":
        # load the model
        print("Pefroming Rollout...")
        trainer.model.load_state_dict(
            torch.load(f"./experiments/{args.ensemble_id}_bestStateDict_nll")
        )
        rolloutPred = trainer.rollout(testLoader)
        with open(
            f"./experiments/{args.ensemble_id}_Rollout_nll.pkl", "wb"
        ) as f:
            pickle.dump(rolloutPred, f)
    return


def getBestConfig(df, ensemble_id):
    # remove 'F'
    df = df[df["objective_0"] != "F"]
    # convert to float and take the negative
    df.loc[:, 'objective_0'] = df['objective_0'].values.astype(float)
    df.loc[:, 'objective_1'] = df['objective_1'].values.astype(float)
    # Pick the best objectives

    max_row_index = (
        (df["objective_0"] + df["objective_1"])
        .sort_values(ascending=False)
        .index[ensemble_id]
    )
    df = df.rename(columns=lambda x: x.replace("p:", ""))
    obj_0 = df.loc[max_row_index, 'objective_0']
    obj_1 = df.loc[max_row_index, 'objective_1']
    print(f"Config {max_row_index}, Objective 0 {obj_0} Objective 1 {obj_1}")
    return df.loc[max_row_index]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--df_path",
        type=str,
        default="./hpo_logs/hpo_nll_quantile_new/results.csv",
    )
    parser.add_argument("--ensemble_id", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(
        args.df_path,
    )
    config = dict(getBestConfig(df, args.ensemble_id))
    run(config, args)
