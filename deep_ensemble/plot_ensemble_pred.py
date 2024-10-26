import pickle

import numpy as np
from metrics import NMAE, NMSE, r2_score
from plot import (plot_num_members_metrics, plot_rollout_metrics,
                  plot_single_field, plot_uncertainty_field)

# load data
data_dir = "/Users/yixuan.sun/Downloads/hpo_uq_pred/"

data = np.load(data_dir + "topK_test_ensemble_predictions.npz")

true_labels = data["true_labels"]
pred_mean = data["pred_mean"]
aleatoric_uc = data["aleatoric_uc"]
epistemic_uc = data["epistemic_uc"]

sample_idx = 0
var_id = 1
mask = true_labels[sample_idx, var_id] == 0
var_names = [
    "Salinity",
    "Temperature",
    "Meridional Velocity",
    "Zonal Velocity",
    "GM",
]

with open(data_dir + "SOMA_mask.pkl", "rb") as f:
    mask = pickle.load(f)
mask1 = mask["mask1"]
mask2 = mask["mask2"]
mask = np.logical_or(mask1, mask2)[0, 0, :, :, 0]

# plot true field
print(true_labels.shape)
fig = plot_single_field(
    field=true_labels[sample_idx, var_id],
    field_name=var_names[var_id],
    vmin=None,
    vmax=None,
    mask=mask,
)
fig.savefig(
    f"./figs/ensemble_pred/true_field_{var_names[var_id]}.pdf",
    format="pdf",
    bbox_inches="tight",
)

# plot mean prediction
fig = plot_single_field(
    field=pred_mean[sample_idx, var_id],
    field_name=var_names[var_id],
    vmin=None,
    vmax=None,
    mask=mask,
)
fig.savefig(
    f"./figs/ensemble_pred/pred_mean_{var_names[var_id]}.pdf",
    format="pdf",
    bbox_inches="tight",
)

# plot aleatoric uncertainty
fig = plot_uncertainty_field(
    field=aleatoric_uc[sample_idx, var_id],
    field_name=var_names[var_id],
    vmin=None,
    vmax=None,
    mask=mask,
)
fig.savefig(
    f"./figs/ensemble_pred/aleatoric_uc_{var_names[var_id]}.pdf",
    format="pdf",
    bbox_inches="tight",
)

# plot epistemic uncertainty
fig = plot_uncertainty_field(
    field=epistemic_uc[sample_idx, var_id],
    field_name=var_names[var_id],
    vmin=None,
    vmax=None,
    mask=mask,
)
fig.savefig(
    f"./figs/ensemble_pred/epistemic_uc_{var_names[var_id]}.pdf",
    format="pdf",
    bbox_inches="tight",
)
# plot rollout
rollouts = np.load(data_dir + "topK_test_rollout.npz")
true = rollouts["true"]
pred = rollouts["pred"]
print(true.shape)
print(pred.shape)

fig = plot_rollout_metrics(true, pred, mask, score_fn=NMSE)
fig.savefig(
    f"./figs/ensemble_pred/rollout_NMSE.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig = plot_rollout_metrics(true, pred, mask, score_fn=NMAE)
fig.savefig(
    f"./figs/ensemble_pred/rollout_NMAE.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig = plot_rollout_metrics(true, pred, mask, score_fn=r2_score)
fig.savefig(
    f"./figs/ensemble_pred/rollout_r2_score.pdf",
    format="pdf",
    bbox_inches="tight",
)

# plot ensemble members vs scores
en_mem_pred = np.load(data_dir + "topK_val_pred_vs_num_members.npz")
pred_mean = en_mem_pred["pred_mean"]
aleatoric_uc = en_mem_pred["aleatoric_uc"]
epistemic_uc = en_mem_pred["epistemic_uc"]
true = en_mem_pred["true"]

print(pred_mean.shape, aleatoric_uc.shape, epistemic_uc.shape, true.shape)

fig = plot_num_members_metrics(
    true=true,
    pred_mean=pred_mean,
    pred_std=np.sqrt(aleatoric_uc),
    score_fn=NMSE,
)
fig.savefig(
    f"./figs/ensemble_pred/num_members_NMSE.pdf",
    format="pdf",
    bbox_inches="tight",
)
