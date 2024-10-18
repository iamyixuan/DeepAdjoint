import argparse
import pickle

import numpy as np
from metrics import MAE, MSE, NMAE, NMSE, RMSE, r2_score
from plot import map_to_physical_scale

parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="nll")
parser.add_argument("--var_id", type=int, default=1)
args = parser.parse_args()
root_dir = "./experiments/"
ensemble_pred = []
sample_id = 0
for i in range(10):
    with np.load(f"{root_dir}{i}_Pred_{args.loss}.npz") as data:
        pred = data["pred"]
        true = data["true"]
        # correct the true pred
        if pred.shape[1] < true.shape[1]:
            temp = pred
            pred = true
            true = temp
        ch = true.shape[1]
        if args.loss == "nll":
            pred = pred[:, [args.var_id, args.var_id + ch]]
        elif args.loss == "quantile":
            idx = [args.var_id, args.var_id + ch, args.var_id + 2 * ch]
            pred = pred[:, idx]
        else:
            pred = pred[:, [args.var_id]]

        mask = true[sample_id, args.var_id] == 0
        true = map_to_physical_scale(
            true[:, args.var_id], args.var_id, args.loss
        )

        if args.loss == "mse" or args.loss == "mae":
            pred = map_to_physical_scale(
                pred,
                args.var_id,
                args.loss,
            )
        else:
            pred = map_to_physical_scale(
                pred, args.var_id, args.loss, if_pred=True
            )

        if args.loss == "quantile":
            pred = pred[:, 1]
        else:
            pred = pred[:, 0]
        ensemble_pred.append(pred)

ensemble_pred = np.array(ensemble_pred).mean(axis=0)
print(true.shape, ensemble_pred.shape)

MSE_score = MSE(true[:, ~mask], ensemble_pred[:, ~mask])
MAE_score = MAE(true[:, ~mask], ensemble_pred[:, ~mask])
RMSE_score = RMSE(true[:, ~mask], ensemble_pred[:, ~mask])
NMAE_score = NMAE(true[:, ~mask], ensemble_pred[:, ~mask])
NMSE_score = NMSE(true[:, ~mask], ensemble_pred[:, ~mask])
r2_score = r2_score(true[:, ~mask], ensemble_pred[:, ~mask])

print(f"var_id: {args.var_id}")
print(f"MSE: {MSE_score}")
print(f"MAE: {MAE_score}")
print(f"NMSE: {NMSE_score}")
print(f"NMAE: {NMAE_score}")
print(f"RMSE: {RMSE_score}")
print(f"r2_score: {r2_score}")
