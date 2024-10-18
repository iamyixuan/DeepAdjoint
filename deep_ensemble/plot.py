import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from metrics import MAE, MSE, NMAE, NMSE, RMSE, r2_score

# set figure size


def set_size(width, fraction=1, subplots=(1, 1), golden_ratio=True):
    """Set figure dimensions to avoid scaling in LaTeX.
    from https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "sans-serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
    }

    plt.rcParams.update(tex_fonts)

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if golden_ratio:
        fig_height_in = (
            fig_width_in * golden_ratio * (subplots[0] / subplots[1])
        )
    else:
        fig_height_in = fig_width_in

    return (fig_width_in, fig_height_in)


# map it back to original scale
def map_to_physical_scale(x, var_id, loss="nll", if_pred=False):
    datamin = np.array([34.01481, 5.144762, 3.82e-8, 6.95e-9, 200])
    datamax = np.array([34.24358, 18.84177, 0.906503, 1.640676, 2000])
    if if_pred:
        if loss == "nll":
            assert x.shape[1] == 2
            mean = x[:, 0]
            var = x[:, 1]

            rescale_mean = (
                mean * (datamax[var_id] - datamin[var_id]) + datamin[var_id]
            )
            rescale_var = var * (datamax[var_id] - datamin[var_id]) ** 2

            x = np.stack([rescale_mean, rescale_var], axis=1)
            return x
        elif loss == "quantile":
            assert x.shape[1] == 3
            return x * (datamax[var_id] - datamin[var_id]) + datamin[var_id]
        else:
            return x * (datamax[var_id] - datamin[var_id]) + datamin[var_id]
    else:
        return x * (datamax[var_id] - datamin[var_id]) + datamin[var_id]


# plot training curve
def plot_train_curve(train_loss, val_loss, fig_width="thesis"):
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    ax.plot(train_loss, label="Train Loss")
    ax.plot(val_loss, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_train_curve_ensemble(
    train_loss_list, val_loss_list, fig_width="thesis"
):
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    train_loss_list = np.array(train_loss_list)
    train_loss_mean = np.median(train_loss_list, axis=0)

    val_loss_mean = np.median(val_loss_list, axis=0)
    train_loss_max = np.quantile(train_loss_list, 0.9, axis=0)
    train_loss_min = np.quantile(train_loss_list, 0.1, axis=0)
    val_loss_max = np.max(val_loss_list, axis=0)
    val_loss_min = np.min(val_loss_list, axis=0)
    ax.plot(train_loss_mean, color="blue", label="Train Loss")
    ax.fill_between(
        range(len(train_loss_mean)),
        train_loss_min,
        train_loss_max,
        color="blue",
        alpha=0.15,
    )
    ax.plot(val_loss_mean, color="r", label="Val Loss")
    ax.fill_between(
        range(len(val_loss_mean)),
        val_loss_min,
        val_loss_max,
        color="red",
        alpha=0.15,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ymax = np.max(train_loss_mean)
    ymin = np.min(train_loss_mean)
    if ymin > 0:
        ax.set_yscale("log")
    else:
        ymin = ymin - 0.05 * (ymax - ymin)
        ymax = ymax + 0.05 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)
    ax.grid(linestyle="dotted")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_score_ensemble_compare(
    score_list, loss_names, score_name, fig_width="thesis"
):
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    if score_name == r"$1-R^2$":
        score_list = 1 - np.array(score_list)
    else:
        score_list = np.array(score_list)

    for i in range(len(score_list)):
        assert len(score_list[i]) == 10, f"num of scores {len(score_list[i])}"
        score_median = np.median(score_list[i], axis=0)
        score_max = np.quantile(score_list[i], 0.9, axis=0)
        score_min = np.quantile(score_list[i], 0.1, axis=0)
        ax.plot(score_median, label=f"{loss_names[i]}", color=f"C{i}")
        ax.fill_between(
            range(len(score_median)),
            score_min,
            score_max,
            color=f"C{i}",
            alpha=0.15,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{score_name}")
    # ymax = np.quantile(score_mean, 0.99)
    # ymin = np.min(train_score_mean)
    # ymin = ymin - 0.05 * (ymax - ymin)
    # ax.set_ylim(ymin, ymax)
    # if score_name == "R2":
    #     ax.set_ylim(0.5, 1.05)
    # else:
    ax.set_yscale("log")
    ax.grid(linestyle="dotted")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_score_num_members(
    score_list, loss_names, score_name, fig_width="thesis"
):
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    if score_name == r"$1-R^2$":
        score_list = 1 - np.array(score_list)
    else:
        score_list = np.array(score_list)
    end_score = score_list[..., -1]

    for i in range(len(loss_names)):
        median_by_member = []
        tenper_by_member = []
        ninetper_by_member = []

        assert end_score.shape[1] == 10
        print("shape of end_score", end_score.shape)
        for j in range(end_score.shape[1]):
            median_by_member.append(np.median(end_score[i, : j + 1]))
            tenper_by_member.append(np.quantile(end_score[i, : j + 1], 0.1))
            ninetper_by_member.append(np.quantile(end_score[i, : j + 1], 0.9))
        ax.plot(median_by_member, label=f"{loss_names[i]}", color=f"C{i}")
        ax.fill_between(
            range(len(median_by_member)),
            tenper_by_member,
            ninetper_by_member,
            color=f"C{i}",
            alpha=0.15,
        )

    ax.grid(linestyle="dotted")
    ax.legend()
    ax.set_xticks(range(0, 10))
    ax.set_xticklabels([str(i + 1) for i in range(10)])
    ax.set_xlabel("Number of Members")
    ax.set_ylabel(f"{score_name}")
    # if score_name == "R2":
    #     ax.set_ylim(0.5, 1.05)
    # else:
    ax.set_yscale("log")
    return fig


# plot single-step prediciton
def plot_single_field(
    field, field_name, vmin, vmax, mask=None, fig_width="thesis"
):
    if mask is not None:
        field[mask] = np.nan

    fig, ax = plt.subplots(
        1, 1, figsize=set_size(fig_width, golden_ratio=False)
    )
    im = ax.imshow(field, cmap="plasma", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(field_name)
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    fig.colorbar(im, cax=cax)
    plt.tight_layout()
    return fig


# plot rollout
def plot_rollout_metrics(true, pred, mask, score_fn, fig_width="thesis"):
    var_names = [
        "Salinity",
        "Temperature",
        "Meridional Velocity",
        "Zonal Velocity",
    ]
    T = true.shape[0]
    scores = []
    for t in range(T):
        cur_scores = []
        for var_id in range(true.shape[1]):
            true_ = true[t, var_id][~mask]
            pred_ = pred[t, var_id][~mask]
            cur_scores.append(score_fn(true_, pred_))
        scores.append(cur_scores)
    scores = np.array(scores)  # [T, ch]
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    for var_id in range(true.shape[1]):
        ax.plot(scores[:, var_id], label=f"{var_names[var_id]}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_rollout_compare(
    true_list,
    pred_list,
    loss_names,
    var_id,
    mask,
    score_fn,
    fig_width="thesis",
):
    var_names = [
        "Salinity",
        "Temperature",
        "Meridional Velocity",
        "Zonal Velocity",
    ]
    true_list = [map_to_physical_scale(x, var_id) for x in true_list]
    pred_list = [map_to_physical_scale(x, var_id) for x in pred_list]

    T = true_list[0].shape[0]
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    # Define a cycle of line styles
    line_styles = cycler("linestyle", ["-", "-.", "--", "dotted"])

    # Set the line style cycle for this axes
    ax.set_prop_cycle(line_styles)
    for i, (pred, true) in enumerate(zip(pred_list, true_list)):
        scores = []
        for t in range(T):
            cur_scores = []
            true_ = true[t, var_id][~mask]
            pred_ = pred[t, var_id][~mask]
            cur_scores.append(score_fn(true_, pred_))
            scores.append(cur_scores)
        scores = np.array(scores)  # [T, ch]
        ax.plot(scores, label=f"{loss_names[i]}", color=f"C{i}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Score")
    ax.set_title(f"{var_names[var_id]} rollout")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(linestyle="dotted")
    plt.tight_layout()
    return fig


# plot the uncertainty
def plot_uncertainty_field(
    field, field_name, vmin, vmax, mask=None, fig_width="thesis"
):
    if mask is not None:
        field[mask] = np.nan

    if vmax is None:
        vmax = np.nanquantile(field, 0.90)

    fig, ax = plt.subplots(
        1, 1, figsize=set_size(fig_width, golden_ratio=False)
    )
    im = ax.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(field_name)
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    fig.colorbar(im, cax=cax)
    plt.tight_layout()
    return fig


# plot some metrics scores


def plot_metrics(metrics_field, fig_width="thesis"):
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    ax.imshow(metrics_field, cmap="RdBu_r")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    import pickle
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="nll")
    parser.add_argument("--var_id", type=int, default=1)
    args = parser.parse_args()

    # load the training curve
    root_dir = "./experiments/"
    fig_folder = "noisy_ensemble"

    if os.path.exists(f"./figs/{fig_folder}") is False:
        os.mkdir(f"./figs/{fig_folder}")


    train_loss_list = []
    val_loss_list = []
    curve_len = []
    # scores = {
    #     "Val_r2": [],
    #     "Val_RMSE": [],
    #     "Val_MAE": [],
    #     "Val_NMAE": [],
    #     "Val_NMSE": [],
    # }
    for ensemble_id in range(10):
        logname = f"{ensemble_id}_train_curve_{args.loss}.pkl"
        with open(f"{root_dir}{logname}", "rb") as f:
            train_curve = pickle.load(f)

        train_loss = train_curve["TrainLoss"]
        val_loss = train_curve["ValLoss"]
        print("trainloss", train_loss[0], train_loss[-1])
        print("valloss", val_loss[0], val_loss[-1])
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        curve_len.append(len(train_loss))
        # scores["Val_r2"].append(train_curve["Val_r2"])
        # scores["Val_RMSE"].append(train_curve["Val_RMSE"])
        # scores["Val_MAE"].append(train_curve["Val_MAE"])
        # scores["Val_NMAE"].append(train_curve["Val_NMAE"])
        # scores["Val_NMSE"].append(train_curve["Val_NMSE"])

    min_len = int(np.min(curve_len))
    print("min_len", min_len)
    train_loss_list = [x[:min_len] for x in train_loss_list]
    val_loss_list = [x[:min_len] for x in val_loss_list]

    fig = plot_train_curve_ensemble(train_loss_list, val_loss_list)
    fig.savefig(
        f"./figs/{fig_folder}/ensemble_train_curve_{args.loss}.pdf",
        bbox_inches="tight",
    )
    #
    # fig = plot_score_ensemble(scores, ["r2", "RMSE", "MAE", "NMAE", "NMSE"])
    # fig.savefig(
    #     f"./figs/val_score_curve_{args.loss}.pdf",
    #     bbox_inches="tight",
    # )

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
                pred = pred[:, args.var_id]

            mask = true[sample_id, args.var_id] == 0
            true = map_to_physical_scale(
                true[:, args.var_id], args.var_id, args.loss
            )
            pred = map_to_physical_scale(
                pred, args.var_id, args.loss, if_pred=True
            )

            ensemble_pred.append(pred)

    vmin = np.quantile(true[sample_id][~mask], 0.1)
    vmax = np.quantile(true[sample_id][~mask], 0.9)
    print("vmin", vmin, "vmax", vmax)

    if args.loss == "nll":
        ensemble_pred = np.array(ensemble_pred)
        pred = np.mean(ensemble_pred, axis=0)[:, 0]

        al_uc = np.mean(ensemble_pred, axis=0)[sample_id, 1]
        ep_uc = np.var(ensemble_pred, axis=0, ddof=1)[sample_id, 0]
    elif args.loss == "quantile":
        ensemble_pred = np.array(ensemble_pred)
        pred = np.mean(ensemble_pred, axis=0)[:, 1]
        qt_sq = ((ensemble_pred[:, :, 2] - ensemble_pred[:, :, 0]) / 2) ** 2
        al_uc = np.mean(qt_sq, axis=0)[sample_id]
        ep_uc = np.var(ensemble_pred, axis=0, ddof=1)[sample_id, 1]
    else:
        ensemble_pred = np.array(ensemble_pred)
        pred = np.mean(ensemble_pred, axis=0)
        ep_uc = np.var(ensemble_pred, axis=0, ddof=1)[sample_id]
        al_uc = np.zeros_like(ep_uc)

    vmin_uc = None  # np.min([al_uc, ep_uc])
    vmax_uc = None
    var_names = [
        "Salinity",
        "Temperature",
        "Meridional Velocity",
        "Zonal Velocity",
        "GM",
    ]

    pred_field_fig = plot_single_field(
        pred[sample_id], var_names[args.var_id], vmin, vmax, mask=mask
    )
    pred_field_fig.savefig(
        f"./figs/{fig_folder}/pred_field_{args.loss}_{args.var_id}_{sample_id}.pdf",
        bbox_inches="tight",
    )
    true_field_fig = plot_single_field(
        true[sample_id], var_names[args.var_id], vmin, vmax, mask=mask
    )
    true_field_fig.savefig(
        f"./figs/{fig_folder}/true_field_{args.loss}_{args.var_id}_{sample_id}.pdf",
        bbox_inches="tight",
    )

    al_uc_fig = plot_uncertainty_field(
        al_uc,
        f"Aleatoric Uncertainty - {var_names[args.var_id]}",
        vmin_uc,
        vmax_uc,
        mask=mask,
    )
    al_uc_fig.savefig(
        f"./figs/{fig_folder}/aleatoric_uncertainty_{args.loss}_{args.var_id}_{sample_id}.pdf",
        bbox_inches="tight",
    )
    ep_uc_fig = plot_uncertainty_field(
        ep_uc,
        f"Epistemic Uncertainty - {var_names[args.var_id]}",
        vmin=0,
        vmax=vmax_uc,
        mask=mask,
    )
    ep_uc_fig.savefig(
        f"./figs/{fig_folder}/epistemic_uncertainty_{args.loss}_{args.var_id}_{sample_id}.pdf",
        bbox_inches="tight",
    )

    # plot the metrics
    # load rollout
    # rollout = np.load(
    #     f"./experiments/ensemble_rollout_{args.loss}.npz",
    # )
    # true = rollout["true"]
    # pred = rollout["pred"]
    # al_uc = rollout["al_uc"]
    # ep_uc = rollout["ep_uc"]
    # rollout_rMSE = plot_rollout_metrics(
    #     true, pred, mask, MSE, fig_width="thesis"
    # )
    # rollout_rMSE.savefig(
    #     f"./figs/rollout_MSE_{args.loss}.pdf",
    #     bbox_inches="tight",
    # )
    # rollout_rMAE = plot_rollout_metrics(
    #     true, pred, mask, MAE, fig_width="thesis"
    # )
    # rollout_rMAE.savefig(
    #     f"./figs/rollout_MAE_{args.loss}.pdf",
    #     bbox_inches="tight",
    # )
    # rollout_r2 = plot_rollout_metrics(
    #     true, pred, mask, r2_score, fig_width="thesis"
    # )
    # rollout_r2.savefig(
    #     f"./figs/rollout_r2_{args.loss}.pdf",
    #     bbox_inches="tight",
    # )
    # rollout_NMAE = plot_rollout_metrics(
    #     true, pred, mask, NMAE, fig_width="thesis"
    # )
    # rollout_NMAE.savefig(
    #     f"./figs/rollout_NMAE_{args.loss}.pdf",
    #     bbox_inches="tight",
    # )
    # rollout_NMSE = plot_rollout_metrics(
    #     true, pred, mask, NMSE, fig_width="thesis"
    # )
    # rollout_NMSE.savefig(
    #     f"./figs/rollout_NMSE_{args.loss}.pdf",
    #     bbox_inches="tight",
    # )
