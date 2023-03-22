import jax
import numpy as np
import jax.numpy as jnp
import optax
import pickle
import os
from datetime import datetime
from jax.config import config
from functools import partial
from jax import vmap, random
from utils.scaler import MinMaxScaler
from utils.metrics import r2

class MLP:
    def __init__(self, layers, in_dim, out_dim, act_fn) -> None:
        self.layers = [in_dim] + layers + [out_dim]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = self.init_network(random.PRNGKey(5))
        self.act_fn = act_fn

    def init_network(self, key):
        initializer = jax.nn.initializers.glorot_normal()
        keys = random.split(key, len(self.layers))
        def init_layer(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return initializer(w_key, (n, m)), random.normal(b_key, (n,))
        return [init_layer(m, n, k) for m, n, k in zip(self.layers[:-1], self.layers[1:], keys)]

    def activation(self, x):
        if self.act_fn == 'relu':
            return jax.nn.relu(x)
        elif self.act_fn == 'tanh':
            return jnp.tanh(x)
        elif self.act_fn == 'sigmoid':
            return jax.nn.sigmoid(x)
        elif self.act_fn == 'gelu':
            return jax.nn.gelu(x)
        else:
            return x
    
    def forward(self, params, x):
        inputs = x
        for w, b in params[:-1]:
            inputs = jnp.dot(w, inputs) + b
            inputs = self.activation(inputs)
        w_f, b_f = params[-1]
        out = jnp.dot(w_f, inputs) + b_f
        return out
    
    def apply(self, params, x):
        f_pass_v = vmap(self.forward, in_axes=(None, 0))
        return f_pass_v(params, x)
    
    def nn_adjoint(self, params, x):
        def adjoint(params, x):
            jac = jax.jacfwd(self.forward, argnums=1)(params, x)
            return jac
        return vmap(adjoint, in_axes=(None, 0), out_axes=0)(params, x)

class Trainer:
    def __init__(self, net, num_epochs, batch_size, learning_rate, optimizer, scaler=None):
        self.net = net
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.scaler = scaler
        

    def loss(self, params, x, y, adj_y, alpha):
        pred = self.net.apply(params, x)
        adj = self.net.nn_adjoint(params, x)
        adj_loss = jnp.mean((adj - adj_y)**2)
        totLoss = jnp.mean((pred - y)**2) + alpha*adj_loss
        return totLoss, adj_loss

    @partial(jax.jit, static_argnums=(0,)) 
    def step_(self, params, x, y, adj_y, alpha, opt_state):
        # ls, adj_loss = self.loss(params, x, y, adj_y, alpha)
        (ls, adj_loss), grads = jax.value_and_grad(self.loss, argnums=0, has_aux=True)(params, x, y, adj_y, alpha)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, ls, opt_state, adj_loss


    def train_(self, params, train_data, val_data, alpha, save_date=None):
        x_train = train_data['x']
        adj_train = train_data['adj']
        y_train = train_data['y']


        x_val = val_data['x']
        adj_val = val_data['adj']
        y_val = val_data['y']

        logger = {'train_loss': [], 
                  'train_adj_loss':[],
                  'val_loss': [], 
                  'val_adj_loss':[],
                  'train_r2': [], 
                  'val_r2':[]}

        if self.scaler is not None:
            x_scaler = self.scaler(x_train)
            adj_scaler = self.scaler(adj_train)
            y_scaler = self.scaler(y_train)

            # x_train = x_scaler.transform(x_train)
            adj_train = adj_scaler.transform(adj_train)
            y_train = y_scaler.transform(y_train)

            # x_val = x_scaler.transform(x_val)
            adj_val = adj_scaler.transform(adj_val)
            y_val = y_scaler.transform(y_val)

            logger['x_scaler'] = x_scaler
            logger['adj_scaler'] = adj_scaler
            logger['y_scaler'] = y_scaler
    
        
        opt_state = self.optimizer.init(params)

        best_val = jnp.inf
        patience = 1000
        tol = 0
        running_alpha = 0
        for ep in range(self.num_epochs):
            print("alpha is ", running_alpha)
            train_running_ls = []
            train_running_r2 = []
            train_running_adj_ls = []
            if (ep+1) % 10 == 0:# and running_alpha <= alpha:
                running_alpha += 1 #alpha/10.
            for i in range(len(x_train)//self.batch_size):
                x_batch = x_train[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = y_train[i*self.batch_size:(i+1)*self.batch_size]
                adj_batch = adj_train[i*self.batch_size:(i+1)*self.batch_size]
                params, ls, opt_state, adj_ls_batch= self.step_(params, x_batch, y_batch,  adj_batch, running_alpha, opt_state)
                pred_batch = self.net.apply(params, x_batch)
                train_running_r2.append(r2(y_batch, pred_batch))
                train_running_ls.append(ls)
                train_running_adj_ls.append(adj_ls_batch)

            ls_val, val_adj_loss = self.loss(params, x_val, y_val, adj_val, alpha) # set alpha with the preset value in validation. 
            pred_val = self.net.apply(params, x_val)
            logger['train_loss'].append(jnp.asarray(train_running_ls).mean())
            logger['train_r2'].append(jnp.asarray(train_running_r2).mean())
            logger['train_adj_loss'].append(jnp.asarray(train_running_adj_ls).mean())
            logger['val_loss'].append(ls_val)
            logger['val_adj_loss'].append(val_adj_loss)
            logger['val_r2'].append(r2(y_val, pred_val))
            with open('./logs/logger_'+save_date, 'wb') as f:
                pickle.dump(logger, f)
            print('Epoch: {} training loss {:.4f} validation loss {:.4f}'.format(ep, logger['train_loss'][-1], logger['val_loss'][-1]))
            print('training r2 {:.4f} validation r2 {:.4f}'.format(logger['train_r2'][-1], logger['val_r2'][-1]))
            print('training adj loss {:.4f} validation adj loss {:.4f}'.format(logger['train_adj_loss'][-1], logger['val_adj_loss'][-1]))
            print()
            if ls_val < best_val:
                logger['best_params'] = params
                print('Save the best params at epoch {}'.format(ep))
                best_val = ls_val 
                tol = 0
            tol+=1
            if tol > patience:
                print("Early stopping...")
                break       
        return params    
    
    def predict(self, params, x):
        pred = self.net.apply(params, x)
        return pred



if __name__ == "__main__":
    from utils.data_loader import split_data
    import os
    import argparse

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default=None)
    parser.add_argument('-a', type=float, default=1)
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=0.0005)
    args = parser.parse_args()

    now = datetime.now()
    now_str = now.strftime('%m-%d-%H') + '_' + args.name + '_lr' + str(args.lr) +'_alpha' + str(args.a) 

    # config.update("jax_enable_x64", True)

    data = np.load('../data/vary_beta.dat.npz')
    inputs = data['inputs']
    uout = data['uout']
    jrav = data['jrav']

    train, val, test = split_data(inputs, uout, jrav, shuffle_all=True)
    


    save_name = 'mixed_init'
    net = MLP([256]*10, in_dim=159, out_dim=80, act_fn='relu')
    sup = Trainer(net=net, 
                num_epochs=args.epoch, 
                batch_size=args.batch_size, 
                learning_rate=args.lr, 
                optimizer=optax.adam,
                scaler=None)
    net_params = sup.train_(net.params, train, val, args.a, now_str)
    
    