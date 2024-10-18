import argparse
import os

import numpy as np
from metrics import MAE, MSE, NMAE, NMSE, RMSE, r2_score
from plot import plot_rollout_compare


def one_minus_r2_score(y_true, y_pred):
    return 1 - r2_score(y_true, y_pred)

parser = argparse.ArgumentParser()
parser.add_argument("--var_id", type=int, default=1)
args = parser.parse_args()

# load the training curve
root_dir = "./experiments/20240929-NLL-QR-MSE-MAE/"
fig_folder = "noiseless_ensemble"
if os.path.exists(f"./figs/{fig_folder}") is False:
    os.mkdir(f"./figs/{fig_folder}")

losses = ["nll", "quantile", 'mae', 'mse']
loss_names = ["NLL", "Quantile", "MAE", "MSE"]
true_list = []
pred_list = []

for loss in losses:
    rollout = np.load(
        f"{root_dir}/ensemble_rollout_{loss}.npz",
    )

    true = rollout["true"]
    pred = rollout["pred"]
    al_uc = rollout["al_uc"]
    ep_uc = rollout["ep_uc"]
    true_list.append(true)
    pred_list.append(pred)

mask = true[0, args.var_id] == 0

fig_MAE = plot_rollout_compare(
    true_list, pred_list, loss_names, args.var_id, mask, MAE
)
fig_MSE = plot_rollout_compare(
    true_list, pred_list, loss_names, args.var_id, mask, MSE
)
fig_RMSE = plot_rollout_compare(
    true_list, pred_list, loss_names, args.var_id, mask, RMSE
)
fig_NMAE = plot_rollout_compare(
    true_list, pred_list, loss_names, args.var_id, mask, NMAE
)
fig_NMSE = plot_rollout_compare(
    true_list, pred_list, loss_names, args.var_id, mask, NMSE
)
fig_r2_score = plot_rollout_compare(
    true_list, pred_list, loss_names, args.var_id, mask, one_minus_r2_score
)

fig_MAE.savefig(
    f"./figs/{fig_folder}/ensemble_rollout_MAE_{args.var_id}.pdf",
    bbox_inches="tight",
)
fig_MSE.savefig(
    f"./figs/{fig_folder}/ensemble_rollout_MSE_{args.var_id}.pdf",
    bbox_inches="tight",
)
fig_RMSE.savefig(
    f"./figs/{fig_folder}/ensemble_rollout_RMSE_{args.var_id}.pdf",
    bbox_inches="tight",
)
fig_NMAE.savefig(
    f"./figs/{fig_folder}/ensemble_rollout_NMAE_{args.var_id}.pdf",
    bbox_inches="tight",
)
fig_NMSE.savefig(
    f"./figs/{fig_folder}/ensemble_rollout_NMSE_{args.var_id}.pdf",
    bbox_inches="tight",
)
fig_r2_score.savefig(
    f"./figs/{fig_folder}/ensemble_rollout_r2_score_{args.var_id}.pdf",
    bbox_inches="tight",
)
