import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from utils import load_burgers_data


with open("./logs/DA_burgerState.pkl", "rb") as f:
    logger = pickle.load(f)


with open("../AdjointMatchingNN/Data/mixed_nu/0.005694-nu.pkl", "rb") as f:
    data = pickle.load(f)
    x, y, adj = load_burgers_data(data)


class Plotter:
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5

    def da_analysis_anim(self, logger, x_true):
        x_bar = np.array(logger["state"])
        print("The analysis state shape is", x_bar.shape)
        nu = x_bar[:, -1]
        d_min = np.min(x_true)
        d_max = np.max(x_true)
        x_bar = x_bar[..., 1:, :-1]

        for i in range(len(x_bar)//10):
            fig, ax = plt.subplots(2, 1, figsize=(10,8))
            ax[0].imshow(x_bar[i*10].T, vmin=d_min, vmax=d_max, origin='lower', extent=[0, 1, -1 , 1], aspect=0.25)
            # ax[0].set_title(r'Reconstructed $\nu={:.5f}$'.format(np.mean(x_bar[i*10][1:, -1])))
            im=ax[1].imshow(x_true.T, vmin=d_min, vmax=d_max, origin='lower', extent=[0, 1, -1 , 1], aspect=0.25)
            # ax[1].set_title(r'True $\nu$=0.005694')
            for a in ax:
                a.set_xlabel(r"$t$")
                a.set_ylabel(r"$x$")
                a.set_box_aspect(0.5)

            bottom, top = 0.1, 0.9
            left, right = 0.1, 0.8
            plt.subplots_adjust(bottom=0.1, right=0.78, top=0.9)
            cax = plt.axes([0.83, 0.1, 0.03, 0.8])
            fig.colorbar(im, cax=cax)
            fig.savefig('./frames/frame-' + str(i) + '.pdf', format='pdf', bbox_inches='tight')
            plt.close()

        # frames = []

        # for i in range(len(x_bar)//10):
        #     field = ax[0].imshow(x_bar[i*10].T, vmin=d_min, vmax=d_max, origin='lower', extent=[0, 1, -1 , 1], aspect=0.25)
        #     frames.append([field])

        # fig.tight_layout()
        # anim = ArtistAnimation(fig, frames)
        # anim.save('burgersDAsimpleGD.gif', writer='pillow')
        # # Show the animation
        # plt.show()

    def solution_animation(self, solutions):
        print("The number of frames is", len(solutions))
        fig, ax = plt.subplots()
        x = np.arange(solutions[0].shape[1] - 1)
        ax.plot(x, input[0, :-1])
        # plt.show()

        # create a line plot with random data
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u$")

        # Create a list to hold the frames of the animation
        frames = []

        # Loop over the data and update the line on each iteration
        for i in range(len(solutions) // 10):
            # Update the data for the line
            (line,) = ax.plot(
                x,
                solutions[i * 10].reshape(
                    -1,
                )[:-1],
            )
            text = ax.text(
                0.5,
                0.8,
                r"$\nu$ = {:.4f} (true value: {:.4f})".format(
                    solutions[i * 10].reshape(
                        -1,
                    )[-1],
                    input[0, -1],
                ),
                transform=ax.transAxes,
            )
            # Add the current frame to the list of frames
            frames.append([line, text])

        anim = ArtistAnimation(fig, frames, interval=5)
        # anim.save('animation1.gif', writer='pillow')
        # Show the animation
        plt.show()

    def domain_plot(self, true, pred):
        fig, ax = plt.subplots(2, 1)
        vmax = np.max(y)
        vmin = np.min(y)
        ax[0].set_box_aspect(0.5)
        ax[0].set_title("True")
        ax[0].imshow(
            y.T,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=[0, 1, 0, 1],
            transform=ax[0].transAxes,
        )
        ax[0].set_axis_off()
        ax[1].set_box_aspect(0.5)
        ax[1].set_axis_off()
        ax[1].set_title("Predicted")
        im = ax[1].imshow(
            pred.T,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=[0, 1, 0, 1],
            transform=ax[1].transAxes,
        )
        fig.tight_layout()
        fig.colorbar(im, ax=ax.ravel().tolist())
        plt.show()


plotter = Plotter()
# plotter.solution_animation(logger["solution"])

plotter.da_analysis_anim(logger=logger, x_true=x)
