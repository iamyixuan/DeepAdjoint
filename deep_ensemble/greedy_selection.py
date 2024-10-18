"""
    This script is to select the ensemble members greedily.
    The process is as follows:
    1. start with the best model, evaluate it on the validation set.
    2. add another model to the ensemble, keep the member based on the ensemlbe performance.
    3. repeat step 2 until reaching the number of ensemble members.
"""

import numpy as np
import torch
from metrics import log_likelihood_score
from train_model_nll import FNO_MU_STD as FNO_nll
from train_model_quantile import FNO_QR as FNO_quantile


def getBestConfig(df, ensemble_id):
    # remove 'F'
    df = df[df["objective_0"] != "F"]
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
    return df.loc[max_row_index]


def ensemble_predict(model_list, valLoader):
    assert len(model_list) >= 2, "Ensemble should have at least 2 models"
    ensemble_preds = []
    for model in model_list:
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valLoader, 0):
                inputs, labels = data
                outputs = model(inputs)
                ensemble_preds.append(outputs.numpy())
    return np.mean(ensemble_preds, axis=0), labels


def load_model(config, path):
    if config["loss"] == "nll":
        net = FNO_nll(
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
    elif config["loss"] == "quantile":
        net = FNO_quantile(
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
    net.load_state_dict(torch.load(path))
    return net


def greedy_selection(hpo_df, ensemble_size, model_dir, valLoader):
    first_config = getBestConfig(hpo_df, 0)
    first_model = load_model(
        first_config, f'{model_dir}/0_bestStateDict_{first_config["loss"]}'
    )

    ensemble = [first_model]
    ensemble_ids = [0]
    available_ids = list(range(1, 20))
    while len(ensemble) < ensemble_size:
        best_score = -np.inf
        best_model = None
        for j in available_ids:
            config = getBestConfig(hpo_df, j)
            model = load_model(config, config["model_path"])
            ensemble.append(model)
            ensemble_preds, true = ensemble_predict(ensemble, valLoader)
            ch = true.shape[1]
            if config["loss"] == "quantile":
                val_pred_mean = ensemble_preds[:, ch : 2 * ch, ...]
                q_16 = ensemble_preds[:, :ch, ...]
                q_84 = ensemble_preds[:, 2 * ch :, ...]
                val_pred_std = np.abs(q_84 - q_16) / 2
            elif config["loss"] == "nll":
                val_pred_mean = ensemble_preds[:, :ch, ...]
                val_pred_std = ensemble_preds[:, ch:, ...]
                val_pred_std = np.sqrt(val_pred_std)

            score = log_likelihood_score(
                loc=val_pred_mean,
                scale=val_pred_std,
                samples=true,
            )
            if score > best_score:
                best_id = j
                best_score = score
                best_model = model
            ensemble.pop()
        ensemble.append(best_model)
        ensemble_ids.append(best_id)
        available_ids.remove(best_id)
    return ensemble, ensemble_ids
