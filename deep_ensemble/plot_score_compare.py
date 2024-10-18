import pickle
import os

import numpy as np
from plot import plot_score_ensemble_compare, plot_score_num_members

root_dir = "./experiments/"

losses = ["nll", "quantile", "mse", "mae"]
MAE = []
RMSE = []
NMAE = []
NMSE = []
R2 = []
MSE = []
for loss in losses:
    train_loss_list = []
    val_loss_list = []
    curve_len = []
    r2_list = []
    rmse_list = []
    mae_list = []
    nmae_list = []
    nmse_list = []
    mse_list = []

    for ensemble_id in range(10):
        logname = f"{ensemble_id}_train_curve_{loss}.pkl"
        with open(f"{root_dir}{logname}", "rb") as f:
            train_curve = pickle.load(f)

        print(train_curve.keys())
        train_loss = train_curve["TrainLoss"]
        val_loss = train_curve["ValLoss"]
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        curve_len.append(len(train_loss))
        r2_list.append(train_curve["Val_r2"])
        rmse_list.append(train_curve["Val_RMSE"])
        mae_list.append(train_curve["Val_MAE"])
        mse_list.append(train_curve["Val_MSE"])
        nmae_list.append(train_curve["Val_NMAE"])
        nmse_list.append(train_curve["Val_NMSE"])

    min_len = int(np.min(curve_len))
    print("loss", loss)
    print("min_len", min_len)
    train_loss_list = np.asarray([x[:min_len] for x in train_loss_list])
    val_loss_list = np.asarray([x[:min_len] for x in val_loss_list])
    r2_list = np.asarray([x[:min_len] for x in r2_list])
    rmse_list = np.asarray([x[:min_len] for x in rmse_list])
    mae_list = np.asarray([x[:min_len] for x in mae_list])
    mse_list = np.asarray([x[:min_len] for x in mse_list])
    print("mse_list", mse_list.shape)
    nmae_list = np.asarray([x[:min_len] for x in nmae_list])
    nmse_list = np.asarray([x[:min_len] for x in nmse_list])

    assert r2_list.shape == (10, min_len)
    assert rmse_list.shape == (10, min_len)
    assert mae_list.shape == (10, min_len)
    assert mse_list.shape == (10, min_len)
    assert nmae_list.shape == (10, min_len)
    assert nmse_list.shape == (10, min_len)

    MAE.append(mae_list)
    MSE.append(mse_list)
    RMSE.append(rmse_list)
    NMAE.append(nmae_list)
    NMSE.append(nmse_list)
    R2.append(r2_list)

fig_folder = "noisy_ensemble"

if os.path.exists(f"./figs/{fig_folder}") is False:
    os.mkdir(f"./figs/{fig_folder}")

loss_names = ["NLL", "Quantile", "MSE", "MAE"]
fig = plot_score_ensemble_compare(MAE, loss_names, "MAE")
fig.savefig(f"./figs/{fig_folder}/MAE_compare.pdf", bbox_inches="tight")
fig = plot_score_ensemble_compare(MSE, loss_names, "MSE")
fig.savefig(f"./figs/{fig_folder}/MSE_compare.pdf", bbox_inches="tight")
fig = plot_score_ensemble_compare(RMSE, loss_names, "RMSE")
fig.savefig(f"./figs/{fig_folder}/RMSE_compare.pdf", bbox_inches="tight")
fig = plot_score_ensemble_compare(NMAE, loss_names, "NMAE")
fig.savefig(f"./figs/{fig_folder}/NMAE_compare.pdf", bbox_inches="tight")
fig = plot_score_ensemble_compare(NMSE, loss_names, "NMSE")
fig.savefig(f"./figs/{fig_folder}/NMSE_compare.pdf", bbox_inches="tight")
fig = plot_score_ensemble_compare(R2, loss_names, r"$1-R^2$")
fig.savefig(f"./figs/{fig_folder}/R2_compare.pdf", bbox_inches="tight")


fig2 = plot_score_num_members(MAE, loss_names, "MAE")
fig2.savefig(f"./figs/{fig_folder}/MAE_num_members.pdf", bbox_inches="tight")
fig2 = plot_score_num_members(MSE, loss_names, "MSE")
fig2.savefig(f"./figs/{fig_folder}/MSE_num_members.pdf", bbox_inches="tight")
fig2 = plot_score_num_members(RMSE, loss_names, "RMSE")
fig2.savefig(f"./figs/{fig_folder}/RMSE_num_members.pdf", bbox_inches="tight")
fig2 = plot_score_num_members(NMAE, loss_names, "NMAE")
fig2.savefig(f"./figs/{fig_folder}/NMAE_num_members.pdf", bbox_inches="tight")
fig2 = plot_score_num_members(NMSE, loss_names, "NMSE")
fig2.savefig(f"./figs/{fig_folder}/NMSE_num_members.pdf", bbox_inches="tight")
fig2 = plot_score_num_members(R2, loss_names, r"$1-R^2$")
fig2.savefig(f"./figs/{fig_folder}/R2_num_members.pdf", bbox_inches="tight")
