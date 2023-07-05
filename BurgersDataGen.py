import numpy as np
import pickle
import argparse
from phi.tf.flow import *
from AdjointMatchingNN.model.adjoint_matching import DifferentiableBurgersSolver


def main(args):
    solver = DifferentiableBurgersSolver(
        init_coeff=args.init,
        NX=args.NX,
        NT=args.NT,
        NU=args.NU,
        XMIN=args.XMIN,
        XMAX=args.XMAX,
        TMAX=args.TMAX,
    )

    sol, grad = solver.get_data()

    with open("./AdjointMatchingNN/Data/mixed_nu/val/" + str(args.NU)[:8] + "-nu.pkl", "wb") as file:
        pickle.dump([np.array(sol), np.array(grad), args.NU], file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-init", type=float, default=1)
    parser.add_argument("-NX", type=int, default=128)
    parser.add_argument("-NT", type=int, default=200)
    parser.add_argument("-NU", type=float, default=0.01 / np.pi)
    parser.add_argument("-XMIN", type=float, default=-1.0)
    parser.add_argument("-XMAX", type=float, default=1.0)
    parser.add_argument("-TMAX", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
