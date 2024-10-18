"""
    Rollout script
    The rollout should be based on the ensemble predictor instead of each member then
    take the average. 
    We need to construct the ensemble predictor and then rollout the sequence.
"""

import numpy as np
import pandas as pd
import torch
from data import SOMAdata
from modulus.models.fno import FNO
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_model import FNO_MU_STD as FNO_NLL
from train_model import getBestConfig
from train_model_quantile import FNO_QR as FNO_QUANTILE


# map it back to original scale
def map_to_physical_scale(x, if_sigma=False):
    datamin = np.array([34.01481, 5.144762, 3.82e-8, 6.95e-9])
    datamax = np.array([34.24358, 18.84177, 0.906503, 1.640676])
    datamin = datamin[None, :, None, None]
    datamax = datamax[None, :, None, None]
    if if_sigma:
        rescale = x * (datamax - datamin) ** 2
        return rescale
    else:
        return x * (datamax - datamin) + datamin


class EnsemblePredictor:
    def __init__(self, config_list, loss="nll"):
        self.members = []
        self.loss = loss
        for id, config in tqdm(enumerate(config_list)):
            if loss == "nll":
                model = FNO_NLL(
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
                model.load_state_dict(
                    torch.load(f"./experiments/{id}_bestStateDict_nll")
                )
                self.members.append(model)
            elif loss == "quantile":
                model = FNO_QUANTILE(
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
                model.load_state_dict(
                    torch.load(f"./experiments/{id}_bestStateDict_quantile")
                )
                self.members.append(model)
            elif loss == "mse" or loss == "mae":
                model = FNO(
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
                model.load_state_dict(
                    torch.load(f"./experiments/{id}_bestStateDict_{loss}")
                )
                self.members.append(model)
            else:
                raise ValueError("Invalid loss function")

    def predict(self, x):
        preds = []
        for member in self.members:
            member.eval()
            member.to(x.device)
            pred = member(x)
            ch = pred.shape[1]
            preds.append(pred)

        if self.loss == "nll":
            assert len(pred.shape) == 4
            assert ch == 4 * 2
            ch = ch // 2

            # shape (num_members, batch, ch * 2, 100, 100)
            ensemble_pred = torch.stack(preds)
            assert ensemble_pred.shape[0] == 10
            pred = torch.mean(ensemble_pred, dim=0)[:, :ch, ...]
            al_uc = torch.mean(ensemble_pred, dim=0)[:, ch:, ...]
            ep_uc = torch.var(ensemble_pred, dim=0)[:, :ch, ...]
        elif self.loss == "quantile":
            assert len(pred.shape) == 4
            assert ch == 3 * 4
            ch = ch // 3
            ensemble_pred = torch.stack(preds)
            assert ensemble_pred.shape[0] == 10
            pred = torch.mean(ensemble_pred[:, :, ch : 2 * ch], dim=0)
            qt_sq = (
                (ensemble_pred[:, :, 2 * ch :] - ensemble_pred[:, :, :ch]) / 2
            ) ** 2
            al_uc = torch.mean(qt_sq, dim=0)
            ep_uc = torch.var(ensemble_pred, dim=0)[:, ch : 2 * ch]
        elif self.loss == "mse" or self.loss == "mae":
            assert ch == 4
            ensemble_pred = torch.stack(preds)
            pred = torch.mean(ensemble_pred, dim=0)
            ep_uc = torch.var(ensemble_pred, dim=0)
            al_uc = torch.zeros_like(ep_uc)
        return pred, al_uc, ep_uc


def rollout(dataloader, loss="nll", df_path="./results/results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(
        df_path,
    )
    config_list = [dict(getBestConfig(df, id)) for id in range(10)]
    ensemble = EnsemblePredictor(config_list, loss)

    # for every 29 steps we save a sequence
    truePred = {"true": [], "pred": [], "al_uc": [], "ep_uc": []}
    temp_pred = []
    temp_al = []
    temp_ep = []
    true = []
    for i, (x, y) in enumerate(dataloader):
        assert x.shape[0] == 1
        print(i)
        x = x.to(device)
        y = y.to(device)
        if i % 29 == 0 and i > 0:
            truePred["true"].append(true)
            truePred["pred"].append(temp_pred)
            truePred["al_uc"].append(temp_al)
            truePred["ep_uc"].append(temp_ep)
            break
            pred, al_uc, ep_uc = ensemble.predict(x)
            temp_pred = [pred]  # reset sequence
            temp_al = [al_uc]
            temp_ep = [ep_uc]
            true = [y]
        else:
            if len(temp_pred) != 0:
                x_cat = torch.cat([temp_pred[-1], x[:, -1:]], dim=1)
                print(x_cat.shape)
                oneStepPred, al_uc, ep_uc = ensemble.predict(x_cat)
                temp_pred.append(oneStepPred)
                temp_al.append(al_uc)
                temp_ep.append(ep_uc)
                true.append(y)
            else:
                pred, al_uc, ep_uc = ensemble.predict(x)
                print(pred.shape, al_uc.shape, ep_uc.shape)
                temp_pred = [pred]
                temp_al = [al_uc]
                temp_ep = [ep_uc]
                true = [y]
    return truePred


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="nll")
    args = parser.parse_args()

    testSet = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "test",
        transform=True,
    )
    testLoader = DataLoader(testSet, batch_size=1)
    truePred = rollout(testLoader, loss=args.loss)

    pred = truePred["pred"][0]
    true = truePred["true"][0]

    pred = np.asarray([p.cpu().detach().numpy() for p in pred]).squeeze()
    true = np.asarray([t.cpu().detach().numpy() for t in true]).squeeze()
    al_uc = np.asarray(
        [a.cpu().detach().numpy() for a in truePred["al_uc"][0]]
    ).squeeze()
    ep_uc = np.asarray(
        [e.cpu().detach().numpy() for e in truePred["ep_uc"][0]]
    ).squeeze()

    pred = map_to_physical_scale(pred)
    true = map_to_physical_scale(true)
    al_uc = map_to_physical_scale(al_uc, if_sigma=True)
    ep_uc = map_to_physical_scale(ep_uc, if_sigma=True)

    np.savez(
        f"./experiments/ensemble_rollout_{args.loss}.npz",
        pred=pred,
        true=true,
        al_uc=al_uc,
        ep_uc=ep_uc,
    )
