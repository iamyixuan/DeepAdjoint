import matplotlib.pyplot as plt
import numpy as np

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
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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


# plot training curve
def plot_train_curve(train_loss, val_loss, fig_width="thesis"):
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    ax.plot(train_loss, label="Train Loss")
    ax.plot(val_loss, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig


def plot_train_curve_ensemble(
    train_loss_list, val_loss_list, fig_width="thesis"
):
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    train_loss_mean = np.mean(train_loss_list, axis=0)
    val_loss_mean = np.mean(val_loss_list, axis=0)
    train_loss_max = np.max(train_loss_list, axis=0)
    train_loss_min = np.min(train_loss_list, axis=0)
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
    ax.legend()
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
    ax.imshow(field, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(field_name)
    return fig


# plot rollout


# plot the uncertainty
def plot_uncertainty_field(
    field, field_name, vmin, vmax, mask=None, fig_width="thesis"
):
    if mask is not None:
        field[mask] = np.nan

    fig, ax = plt.subplots(
        1, 1, figsize=set_size(fig_width, golden_ratio=False)
    )
    ax.imshow(field, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(field_name)
    return fig


# plot some metrics scores


def plot_metrics(metrics_field, fig_width="thesis"):
    fig, ax = plt.subplots(1, 1, figsize=set_size(fig_width))
    ax.imshow(metrics_field, cmap="viridis")
    return fig


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_id", type=int, required=True)
    parser.add_argument("--loss", type=str, default="nll")
    args = parser.parse_args()

    # load the training curve
    root_dir = "./experiments/"

    train_loss_list = []
    val_loss_list = []
    for ensemble_id in range(10):
        if args.loss == "nll":
            logname = f"ensemble{ensemble_id}_log_{args.loss}.pkl"
        else:
            logname = f"{ensemble_id}_train_curve_{args.loss}.pkl"
        with open(f"{root_dir}{logname}", "rb") as f:
            train_curve = pickle.load(f)

        train_loss = train_curve["train"][:200]
        val_loss = train_curve["val"][:200]
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    fig = plot_train_curve_ensemble(train_loss_list, val_loss_list)
    fig.savefig(
        f"./figs/train_curve_{args.ensemble_id}_{args.loss}.pdf",
        bbox_inches="tight",
    )

    # # load the prediction
    # with np.load(f"{root_dir}{args.ensemble_id}_Pred_{args.loss}.npz") as data:
    #     true = data["true"]
    #     pred = data["pred"]
    #
    # ch = true.shape[1]
    # var_id = 0
    # sample_id = 0
    # true = true[sample_id, var_id]
    # if args.loss == "nll":
    #     pred = pred[sample_id, var_id]
    # else:
    #     pred = pred[sample_id, ch + var_id]
    #
    # vmin = np.min(true)
    # vmax = np.max(true)
    # mask = true == 0
    #
    # true_field_fig = plot_single_field(
    #     true, "Temperature", vmin, vmax, mask=mask
    # )
    # true_field_fig.savefig(
    #     f"./figs/true_field_{args.loss}_{var_id}_{sample_id}.pdf",
    #     bbox_inches="tight",
    # )
    # pred_field_fig = plot_single_field(
    #     pred, "Temperature", vmin, vmax, mask=mask
    # )
    # pred_field_fig.savefig(
    #     f"./figs/pred_field_{args.loss}_{var_id}_{sample_id}.pdf",
    #     bbox_inches="tight",
    # )

    # plot ensemlbe prediction and uncertainty

    ensemble_pred = []
    var_id = 0
    sample_id = 0
    for i in range(10):
        with np.load(f"{root_dir}{i}_Pred_{args.loss}.npz") as data:
            true = data["pred"]
            pred = data["true"]
            print(pred.shape, true.shape)
            ch = true.shape[1]
            if args.loss == "nll":
                pred = pred[:, [var_id, var_id + ch]]
            else:
                idx = [var_id, var_id + ch, var_id + 2 * ch]
                pred = pred[:, idx]
            ensemble_pred.append(pred)

    vmin = np.min(true[:, var_id])
    vmax = np.max(true[:, var_id])
    mask = true[sample_id, var_id] == 0

    if args.loss == "nll":
        ensemble_pred = np.array(ensemble_pred)

        al_uc = np.mean(np.array(ensemble_pred), axis=0)[sample_id, 1]
        ep_uc = np.var(np.array(ensemble_pred), axis=0, ddof=1)[sample_id, 0]
    else:
        ensemble_pred = np.array(ensemble_pred)
        pred = np.mean(ensemble_pred[:, :, 1], axis=0)
        qt_sq = ((ensemble_pred[:, :, 2] - ensemble_pred[:, :, 0]) / 2) ** 2
        al_uc = np.mean(qt_sq, axis=0)[sample_id]
        ep_uc = np.var(np.array(ensemble_pred)[:, :, 1], axis=0, ddof=1)[
            sample_id
        ]

    print("pred true shape", pred.shape, true.shape)

    pred_field_fig = plot_single_field(
        pred[sample_id], "Temperature", vmin, vmax, mask=mask
    )
    pred_field_fig.savefig(
        f"./figs/pred_field_{args.loss}_{var_id}_{sample_id}.pdf",
        bbox_inches="tight",
    )
    true_field_fig = plot_single_field(
        true[sample_id, var_id], "Temperature", vmin, vmax, mask=mask
    )
    true_field_fig.savefig(
        f"./figs/true_field_{args.loss}_{var_id}_{sample_id}.pdf",
        bbox_inches="tight",
    )

    al_uc_fig = plot_uncertainty_field(
        al_uc, "Aleatoric Uncertainty", vmin, vmax, mask=mask
    )
    al_uc_fig.savefig(
        f"./figs/aleatoric_uncertainty_{args.loss}_{var_id}_{sample_id}.pdf",
        bbox_inches="tight",
    )
    ep_uc_fig = plot_uncertainty_field(
        ep_uc, "Epistemic Uncertainty", vmin, vmax, mask=mask
    )
    ep_uc_fig.savefig(
        f"./figs/epistemic_uncertainty_{args.loss}_{var_id}_{sample_id}.pdf",
        bbox_inches="tight",
    )
