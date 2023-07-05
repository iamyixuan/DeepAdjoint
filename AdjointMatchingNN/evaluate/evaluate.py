import pickle
import jax
import numpy as np
import optax
import argparse
import matplotlib.pyplot as plt

from VJPMatching import MLP, Trainer
from utils.data_loader import split_data, combine_burgers_data
from utils.metrics import mean_squared_error, r2, mape
from utils.scaler import StandardScaler
from utils.plotter import Plotter


def load_log(filePath):
    with open(filePath, "rb") as f:
        logger = pickle.load(f)
    return logger


def plot_training(logger):
    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(logger["train_loss"], label="train")
    ax[0].plot(logger["val_loss"], label="val")
    # ax[0].set_ylim(0, 1000)
    ax[1].plot(logger["train_adj_loss"], label="train")
    ax[1].plot(logger["val_adj_loss"], label="val")
    ax[2].plot(logger["train_r2"], label="train")
    ax[2].plot(logger["val_r2"], label="val")
    ax[2].set_ylim(0, 1)
    ax[0].set_title("Total loss")
    ax[1].set_title("Adjoint matching loss")
    ax[2].set_title("r2")
    for a in ax:
        a.legend()
        a.set_box_aspect(1)
    # fig.savefig('./figs_adjoint/trainingCurve_mixed_nu.pdf', format='pdf')
    plt.show()


def eval(args, logger):
    if args.problem == "Glacier":
        data = np.load("../data/vary_beta.dat_4-7-23.npz")
        x = data["inputs"]
        y = data["uout"]
        j_beta = data["jac_beta"]
        jrav = data["jac_u"]
        adj = np.concatenate([jrav, j_beta[..., np.newaxis]], axis=-1)
        train, val, test = split_data(x, y, adj, shuffle_all=True)
    elif args.problem == "Burgers":
        x, y, adj = combine_burgers_data("./Data/mixed_nu/")
        train, val, test = split_data(x, y, adj, shuffle_all=True)

    scaler = StandardScaler(train["x"])
    net = MLP(
        [200] * 5,
        in_dim=train["x"].shape[1],
        out_dim=train["y"].shape[1],
        act_fn="tanh",
        scaler=scaler,
    )
    params = logger["final_params"]

    pred_adj = net.full_Jacobian(params, x)
    u_pred = net.apply(params, x)

    def v_prod(v, adj):
        return v @ adj

    v_prod_map = jax.vmap(v_prod, in_axes=(None, 0))
    v = jax.numpy.ones(y.shape[1])

    adj_idx = -2

    print("The forward MSE is {:.4f}".format(mean_squared_error(y, u_pred)))
    print("The forwardR2 is {:.4f}".format(r2(y, u_pred)))
    print("The forward mape is {:.4f}".format(mape(y, u_pred)))
    print(
        "The adj mse is {:.4f}".format(
            mean_squared_error(adj[..., adj_idx:], pred_adj[..., adj_idx:])
        )
    )
    print("The adj R2 is {:4f}".format(r2(adj[..., adj_idx:], pred_adj[..., adj_idx:])))
    print(
        "The adj mape is {:4f}".format(
            mape(adj[..., adj_idx:], pred_adj[..., adj_idx:])
        )
    )
    print(
        "The VJP mse is {:.4f}".format(
            mean_squared_error(v_prod_map(v, adj), v_prod_map(v, pred_adj))
        )
    )
    print(
        "The VJP adj R2 is {:4f}".format(
            r2(v_prod_map(v, adj), v_prod_map(v, pred_adj))
        )
    )
    print(
        "The VJP adj mape is {:4f}".format(
            mape(v_prod_map(v, adj), v_prod_map(v, pred_adj))
        )
    )

    plotter = Plotter()
    fig = plotter.scatter_plot(adj[3, ..., -1], pred_adj[3, ..., -1])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="AdjointMatchingNN")
    parser.add_argument("-a", type=float, default=1)
    parser.add_argument("-epoch", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-lr", type=float, default=0.0001)
    parser.add_argument("-problem", type=str, default="Burgers")
    args = parser.parse_args()

    logger = load_log("./logs/logger_04-20-10_GlacierLast2_lr0.0001_alpha1")

    eval(args, logger)
