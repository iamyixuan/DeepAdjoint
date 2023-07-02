import numpy as np
import tensorflow as tf
from PINN import BurgerSupervisor, get_data, PINN, plotter
from phi.tf.flow import *


class AdjointNN:
    def __init__(self, forward_model):
        self.net = forward_model

    def get_adjoint(self, net_input):
        X = net_input["x_u"]
        x = tf.cast(X[:, 0], tf.float32)
        t = tf.cast(X[:, 1], tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            out = self.net(x, t)
        dm_dx = tape.gradient(out, x)
        dm_dt = tape.gradient(out, t)
        tot_dev = tf.stack([dm_dx, dm_dt], axis=1)
        dm_adj = tf.transpose(tot_dev)
        return dm_adj


class PINN_DP(BurgerSupervisor):
    def __init__(self, nu, net, epochs, lr, alpha, obs, NT, NX, X_U, X_L, T_max, beta):
        super().__init__(nu, net, epochs, lr, alpha)
        self.obs = obs
        self.NT = NT
        self.NX = NX
        self.X_U = X_U
        self.X_L = X_L
        self.T_max = T_max
        self.Nu = nu
        self.DT = T_max / NT

        self.beta = beta  # weight for the resimulation loss
        self.train, self.val, self.test = get_data(
            NT=self.NT,
            NX=self.NX,
            X_U=self.X_U,
            X_L=self.X_L,
            T_max=self.T_max,
            Nu=self.Nu,
        )

    def DP_model(self, N):
        SOLUTION_T16 = CenteredGrid(
            self.obs,
            extrapolation.PERIODIC,
            x=self.NX,
            bounds=Box(x=(self.X_L, self.X_U)),
        )
        return SOLUTION_T16

    def resimulation_loss(self, domain):
        x = domain[:, 0]
        t = domain[:, 1]
        u_pred = self.net(x, t).numpy()
        u_pred = u_pred.reshape(self.NT, self.NX).T
        init_vel = math.tensor(u_pred[:, 0].ravel().tolist(), math.spatial("x"))
        init_vel = CenteredGrid(
            init_vel,
            extrapolation.PERIODIC,
            x=self.NX,
            bounds=Box(x=(self.X_L, self.X_U)),
        )
        velocities = [init_vel]
        for time_step in range(self.NT):
            v1 = diffuse.explicit(1.0 * velocities[-1], self.Nu, self.DT, substeps=1)
            v2 = advect.semi_lagrangian(v1, v1, self.DT)
            velocities.append(v2)
        loss = field.l2_loss(velocities[16] - SOLUTION_T16) * 2.0 / self.NX  # MSE
        return loss.native()

    def train_net(self):
        """override the train method"""
        x_u_train = self.train["x_u"]
        x_f_train = self.train["x_f"]
        y_train = self.train["y_u"]

        x_val = self.val["x"]
        x_u = x_u_train[:, 0]
        t_u = x_u_train[:, 1]

        x_f = x_f_train[:, 0]
        t_f = x_f_train[:, 1]
        best_ls = np.inf

        domain = self.test["x"]
        for i in range(self.epochs):
            train_ls = []
            with tf.GradientTape() as t:
                u_out = self.net(x_u, t_u)
                f_out = self.f_net(x_f, t_f)
                loss_u = self.loss(y_train, u_out)
                loss_f = self.loss(0, f_out)
                loss_r = self.resimulation_loss(domain)
                tot_loss = loss_u + self.alpha * loss_f + self.beta * loss_r
                train_ls.append(tot_loss)
            grads = t.gradient(tot_loss, self.net.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))

            # validation loss
            # val_r2 = self.r2(y_val, self.net(x_val[:, 0], x_val[:, 1]))
            val_f_loss = self.loss(0, self.f_net(x_val[:, 0], x_val[:, 1])).numpy()
            print(
                "Epoch %d, training loss: (%.4f, %.4f), validiation PDE loss %.4f, resimulation loss %.4f"
                % (i + 1, loss_u.numpy(), loss_f.numpy(), val_f_loss, loss_r.numpy())
            )
            if loss_r.numpy() < best_ls:
                self.net.save_weights("./saved_model/PINN_DP_weights_best")
                best_ls = loss_r.numpy()
        return val_f_loss  # return the objective.


if __name__ == "__main__":
    # """
    # test multivariate gradient
    # """
    # x = tf.Variable(2.)
    # y = tf.Variable(3.)

    # with tf.GradientTape(persistent=True) as t:
    #     z = x**2 + y**3
    # out = t.gradient(z, x)
    # out_y = t.gradient(z, y)
    # print(out, out_y)

    N = 200
    DX = 2 / N
    STEPS = 200
    DT = 1 / STEPS
    NU = 0.01 / (N * np.pi)

    # allocate velocity grid
    # velocity = CenteredGrid(0, extrapolation.PERIODIC, x=N, bounds=Box(x=(-1,1)))
    train, val, test = get_data(NT=STEPS, NX=N, X_U=1.0, X_L=-1.0, T_max=1.0, Nu=NU)
    # and a grid with the reference solution
    # REFERENCE_DATA = math.tensor(test['y'].reshape(STEPS, N).T[:, 16].ravel() , math.spatial('x'))
    # SOLUTION_T16 = CenteredGrid(REFERENCE_DATA, extrapolation.PERIODIC, x=N, bounds=Box(x=(-1,1)))

    # training a net
    # net = PINN(num_layers=10, hidden_dim=20, output_dim=1, act_fn='tanh')
    # sup = PINN_DP(nu=NU, net=net, epochs=1000, lr=0.01, alpha=1, obs=REFERENCE_DATA,
    #             NT=STEPS, NX=N, X_U=1., X_L=-1., T_max=1., beta=1)
    # #a = sup.train_net()

    # net = PINN(num_layers=10, hidden_dim=20, output_dim=1, act_fn='tanh')
    net2 = PINN(num_layers=8, hidden_dim=20, output_dim=1, act_fn="tanh")
    # net.load_weights('./saved_model/PINN_DP_weights_best')
    net2.load_weights("./saved_model/PINN_weights_best")
    # sup = PINN_DP(nu=NU, net=net2, epochs=1000, lr=0.01, alpha=1, obs=REFERENCE_DATA,
    #             NT=STEPS, NX=N, X_U=1., X_L=-1., T_max=1., beta=1)
    # # a = sup.train_net()
    plotter(test["x"], net2(test["x"][:, 0], test["x"][:, 1]).numpy(), "pinn", N, STEPS)

    # true = sup.test['y'].reshape(STEPS, N).T
    # pred_pinn = net2(sup.test['x'][:,0], sup.test['x'][:,1]).numpy().reshape(STEPS, N).T
    # pred_pinn_dp = net(sup.test['x'][:,0], sup.test['x'][:,1]).numpy().reshape(STEPS, N).T

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(1,2, figsize=(10,5))
    # ax[0].plot(true[:,0].ravel(), label="True IC")
    # ax[0].plot(pred_pinn[:,0].ravel(), label="PINN Pred")
    # ax[0].plot(pred_pinn_dp[:,0].ravel(), label="PINN-DP")
    # ax[0].set_title("Initial Conditions")
    # ax[1].plot(true[:,16].ravel(), label="True Sol")
    # ax[1].plot(pred_pinn[:,16].ravel(), label="PINN Pred")
    # ax[1].plot(pred_pinn_dp[:,16].ravel(), label="PINN-DP")
    # ax[1].set_title("Step 16")
    # for a in ax:
    #     a.set_xlabel('x')
    #     a.set_ylabel('u')
    #     a.legend()
    # fig.savefig('./init_obs_compare.pdf', format='pdf')
    # plt.show()
