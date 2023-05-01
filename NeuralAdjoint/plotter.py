import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from utils import load_burgers_data


with open('./logs/0_logger.pkl', 'rb') as f:
    logger = pickle.load(f)


with open('../AdjointMatchingNN/Data/mixed_nu/0.001674-nu.pkl', 'rb') as f:
        data = pickle.load(f)
        input, y, adj = load_burgers_data(data)

class Plotter:
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.5

    def solution_animation(self, solutions):
        print('The number of frames is', len(solutions))
        fig, ax = plt.subplots()
        x = np.arange(solutions[0].shape[1] - 1)
        ax.plot(x, input[0,:-1])
        # plt.show()

        #create a line plot with random data
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
    

        # Create a list to hold the frames of the animation
        frames = []

        # Loop over the data and update the line on each iteration
        for i in range(len(solutions)):
            # Update the data for the line
            line, = ax.plot(x, solutions[i].reshape(-1,)[:-1]) 
            text = ax.text(0.5, 0.8, r'$\nu$ = {:.4f} (true value: {:.4f})'.format(solutions[i].reshape(-1,)[-1], input[0, -1]), transform=ax.transAxes)
            # Add the current frame to the list of frames
            frames.append([line, text])

        anim = ArtistAnimation(fig, frames, interval=1)
        # anim.save('animation1.gif', writer='pillow')
        # Show the animation
        plt.show()
    


plotter = Plotter()
plotter.solution_animation(logger['solution'])


