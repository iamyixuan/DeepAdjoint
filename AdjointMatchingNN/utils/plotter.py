import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Plotter:
    def __init__(self):
        plt.rcParams["lines.linewidth"] = 3
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 12

    def heat_map(self, x_t, x_p):
        """
        x: shape (time_steps, x_locs)
        """
        x_t = x_t.T
        x_p = x_p.T
        error = x_t - x_p
        fig, ax = plt.subplots(1, 3, figsize=(12, 8))

        ax[0].imshow(
            x_t,
            cmap="jet",
            interpolation="nearest",
            origin="lower",
            extent=[0, 50, 0, 80],
            aspect="auto",
            vmax=x_t.max(),
            vmin=x_t.min(),
        )
        ax[0].set_xlabel("time steps")
        ax[0].set_ylabel("spatial locs")
        ax[0].set_title("true")

        ax[1].imshow(
            x_p,
            cmap="jet",
            interpolation="nearest",
            origin="lower",
            extent=[0, 50, 0, 80],
            aspect="auto",
            vmax=x_t.max(),
            vmin=x_t.min(),
        )
        ax[1].set_xlabel("time steps")
        ax[1].set_ylabel("spatial locs")
        ax[1].set_title("predicted")

        ax[2].imshow(
            error,
            cmap="jet",
            interpolation="nearest",
            origin="lower",
            extent=[0, 50, 0, 80],
            aspect="auto",
            vmax=x_t.max(),
            vmin=x_t.min(),
        )
        ax[2].set_title("difference")
        plt.show()

    def pca_plot(self, x_t, x_p):
        pca = PCA(n_components=2)
        x_t = pca.fit_transform(x_t)
        x_p = pca.transform(x_p)

        fig, ax = plt.subplots()
        ax.scatter(x_t[:, 0], x_t[:, 1], label="true")
        ax.scatter(x_p[:, 0], x_p[:, 1], label="predicted")
        plt.legend()
        plt.tight_layout()
        return fig

    def scatter_plot(self, true, pred):
        fig, ax = plt.subplots()
        ax.set_box_aspect(1)
        ax.scatter(true, pred)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(color="k", alpha=0.5, linestyle="-.")
        return fig

    def plot_density(self, x):
        fig, axs = plt.subplots(11, 8)
        i = 0
        for ax in axs.flatten():
            if i < 80:
                ax.hist(x[:, i], bins=100)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(left=False, bottom=False)
            i += 1

        return fig

    def save(self, fig, name):
        fig.savefig(
            "../../data/plots/" + name + ".pdf", format="pdf", bbox_inches="tight"
        )


if __name__ == "__main__":
    data = np.load("../../data/vary_ep_glen.dat.npz")
    print(list(data.keys()))
    inputs = data["inputs"]
    uout = data["uout"]
    jrav = data["jrav"]

    plotter = Plotter()
    fig = plotter.plot_density(uout)
    plotter.save(fig, "density-out")
