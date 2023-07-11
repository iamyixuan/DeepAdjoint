import os
import numpy as np
import pickle
import tensorflow as tf
from phi.tf.flow import *


class FNN(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, output_dim, act_fn):
        super().__init__()

        if act_fn == "relu":
            self.act_fn = tf.nn.relu
        elif act_fn == "leaky_relu":
            self.act_fn = tf.nn.leaky_relu
        elif act_fn == "elu":
            self.act_fn = tf.nn.elu
        elif act_fn == "gelu":
            self.act_fn = tf.nn.gelu
        elif act_fn == "tanh":
            self.act_fn = tf.nn.tanh
        elif act_fn == "sigmoid":
            self.act_fn = tf.nn.sigmoid
        else:
            raise ValueError("%s is not in the activation function list" % act_fn)

        self.lys = [tf.keras.layers.Dense(units=hidden_dim, activation=self.act_fn)]
        for n in range(num_layers):
            self.lys.append(
                tf.keras.layers.Dense(units=hidden_dim, activation=self.act_fn)
            )
        self.lys.append(tf.keras.layers.Dense(units=output_dim, activation=None))

    def call(self, u_):
        for layer in self.lys:
            u_ = layer(u_)
        return u_


class DifferentiableBurgersSolver:
    def __init__(self, init_coeff, NX, NT, NU, XMIN, XMAX, TMAX):
        self.NX = NX
        self.NT = NT
        self.NU = np.array([NU])
        self.XMIN = XMIN
        self.XMAX = XMAX
        self.TMAX = TMAX

        self.DX = (XMAX - XMIN) / NX
        self.DT = TMAX / NT

        self.init_cond = lambda x: -np.sin(init_coeff * np.pi * x)
        x = np.linspace(self.XMIN, self.XMAX, self.NX)
        self.cur_u = tf.cast(self.init_cond(x), tf.float32)
        self.cur_u = tf.concat([self.cur_u, self.NU], axis=0)

    def foward_solve(self, cur_u):
        # Nu being the last element in cur_u array.
        # add Nu to the cur_u vector
        Nu = cur_u[-1]
        cur_u = cur_u[:-1]
        cur_u = math.tensor(cur_u, math.spatial("x"))
        cur_u = CenteredGrid(
            cur_u,
            extrapolation.PERIODIC,
            x=self.NX,
            bounds=Box(x=(self.XMIN, self.XMAX)),
        )
        v1 = diffuse.explicit(1.0 * cur_u, Nu, self.DT, substeps=1)
        v2 = advect.semi_lagrangian(v1, v1, self.DT)
        return v2.values.native()

    @tf.function
    def get_gradient(self, cur_u):
        with tf.GradientTape() as tape:
            tape.watch(cur_u)
            v2 = self.foward_solve(cur_u)
        grad = tape.jacobian(v2, cur_u)
        print(grad.shape)
        return v2, grad

    def get_data(self):
        cur_u = self.cur_u
        sol = [cur_u[:-1]]  # only save the states, excluding the viscosity.
        grads = []
        for t in range(self.NT):
            print("the current time step %d" % t)
            v2, grad = self.get_gradient(cur_u)
            sol.append(v2)
            grads.append(grad)
            cur_u = tf.concat([v2, self.NU], axis=0)  # here need to append Nu
        return sol, grads


class AdjointMatchTrainer:
    def __init__(self, net, data_path, save_name):
        self.net = net
        self.data_path = data_path
        self.loss = tf.losses.MeanSquaredError()
        self.save_name = save_name

    def load_data(self, path, mode):
        if mode == "train":
            files = os.listdir(path)[:-1]
        else:
            files = [os.listdir(path)[-1]]
        print(files)
        x_list = []
        y_list = []
        adj_list = []
        for f in files:
            with open(path + f, "rb") as f:
                sol, adj = pickle.load(f)
            sol = np.stack(sol, axis=1)
            adj = np.transpose(np.stack(adj, axis=-1), axes=[2, 0, 1])
            x = sol[..., :-1].T
            y = sol[..., 1:].T
            x_list.append(x)
            y_list.append(y)
            adj_list.append(adj)
        return np.concatenate(x_list), np.concatenate(y_list), np.concatenate(adj_list)

    def prepare_data(self, mode):
        """
        use 60 % of the time to train, 20 validation and rest 20 for testing.
        """
        x_, y_, adj_ = self.load_data(self.data_path, mode)
        rs = np.random.RandomState(0)
        idx = rs.permutation(x_.shape[0])

        x = x_[idx]
        y = y_[idx]
        adj = adj_[idx]

        train_len = int(0.6 * x.shape[0])
        x_train = x[:train_len]
        y_train = y[:train_len]
        adj_train = adj[:train_len]

        x_val = x[train_len:]
        y_val = y[train_len:]
        adj_val = adj[train_len:]

        train_data = {
            "x": tf.cast(x_train, tf.float32),
            "y": tf.cast(y_train, tf.float32),
            "adj": tf.cast(adj_train, tf.float32),
        }
        val_data = {
            "x": tf.cast(x_val, tf.float32),
            "y": tf.cast(y_val, tf.float32),
            "adj": tf.cast(adj_val, tf.float32),
        }
        if mode == "train":
            return train_data, val_data
        else:
            return {
                "x": tf.cast(x_, tf.float32),
                "y": tf.cast(y_, tf.float32),
                "adj": tf.cast(adj_, tf.float32),
            }

    @tf.function
    def obtain_adjoint(self, x):
        """
        Get the adjoints of the neural networks
        """
        x = tf.cast(x, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self.net(x)
            # use batch jacobian to have [batch, in, out] dims.
            adj_nn = tape.batch_jacobian(u, x)
            # adj_nn = tape.gradient(u, x)
        del tape
        return adj_nn

    # @tf.function
    def train_net(self, epochs, learning_rate, alpha):
        logger = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "alpha": alpha,
            "train_std_loss": [],
            "train_adj_loss": [],
            "train_tot_loss": [],
            "val_std_loss": [],
            "val_adj_loss": [],
            "val_tot_loss": [],
        }
        train_data, val_data = self.prepare_data(mode="train")
        x_train = train_data["x"]
        y_train = train_data["y"]
        adj_train = train_data["adj"]
        x_val = val_data["x"]
        y_val = val_data["y"]
        adj_val = val_data["adj"]

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(epochs):
            with tf.GradientTape(persistent=True) as t:
                out = self.net(x_train)
                std_loss = self.loss(y_train, out)
                adj_nn = self.obtain_adjoint(x_train)
                adj_loss = self.loss(adj_train, adj_nn)
                loss_tot = alpha * adj_loss + std_loss  # the total loss
            grads = t.gradient(loss_tot, self.net.trainable_weights)
            # optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
            optimizer.apply_gradients(
                (grad, var)
                for (grad, var) in zip(grads, self.net.trainable_variables)
                if grad is not None
            )

            out_val = self.net(x_val)
            std_val_loss = self.loss(y_val, out_val)
            adj_nn_val = self.obtain_adjoint(x_val)
            adj_val_loss = self.loss(adj_val, adj_nn_val)

            logger["train_std_loss"].append(std_loss.numpy())
            logger["train_adj_loss"].append(adj_loss.numpy())
            logger["train_tot_loss"].append(loss_tot.numpy())
            logger["val_std_loss"].append(std_val_loss.numpy())
            logger["val_adj_loss"].append(adj_val_loss.numpy())
            logger["val_tot_loss"].append((std_val_loss + alpha * adj_val_loss).numpy())
            print(
                "Epoch: {} training standard loss {:.3f} adj loss {:.3f} Val standard loss {:.3f} Adj loss {:.3f}".format(
                    epoch,
                    std_loss.numpy(),
                    adj_loss.numpy(),
                    std_val_loss,
                    adj_val_loss,
                )
            )

            with open("./figs_adjoint/logger_" + self.save_name, "wb") as f:
                pickle.dump(logger, f)

        self.net.save_weights("./figs_adjoint/model/weights_" + self.save_name)


if __name__ == "__main__":
    tf.random.set_seed(0)
    solver = DifferentiableBurgersSolver(0.25, 128, 200, 0.01 / np.pi, -1.0, 1.0, 1.0)
    sol, grad = solver.get_data()
    # import pickle
    # with open('./Data/mixed_init_cond/sol_adj_coef_0.25pi.pkl',  'wb') as f:
    #     pickle.dump([sol, grad], f)

    # with open('./Data/sol_adj.pkl',  'rb') as f:
    #     sol, grad = pickle.load(f)

    # sol = np.stack(sol, axis=1)
    # print(sol.shape)
    # grad = np.stack(grad, axis=-1)
    # save_name = 'mixed_init'

    # net = FNN(num_layers=10, hidden_dim=200, output_dim=128, act_fn='tanh')

    # sup = AdjointMatchTrainer(net=net, data_path='./Data/mixed_init_cond/', save_name=save_name)
    # sup.train_net(850, 0.001, 3)
    # val = sup.prepare_data('val')

    # produce testing: time rollout
    # net.load_weights('./figs_adjoint/weights_mixed_nu')
    # pred = net(val['x']).numpy()
    # pred = np.concatenate([val['x'][0:1], pred])

    # print(pred.shape)
    # x = np.linspace(-1, 1, 128)
    # vel = [-np.sin(np.pi * x).reshape(1, -1)]
    # for i in range(32):
    #     out = sup.net(vel[-1])
    #     vel.append(out)

    # vel = np.concatenate(vel, axis=0).T

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,5))
    # plt.imshow(pred.T, interpolation='nearest',
    #                   cmap='rainbow',
    #                   extent=[0, 1, -1, 1],
    #                   origin='lower', aspect='auto')
    # plt.colorbar()
    # plt.savefig('./figs_adjoint/pred_mixed_nu.pdf', format='pdf')
    # plt.show()
