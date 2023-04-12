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
    def __init__(self, layers, in_dim, out_dim, act_fn, scaler=None) -> None:
        self.layers = [in_dim] + layers + [out_dim]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = self.init_network(random.PRNGKey(5))
        self.act_fn = act_fn
        self.scaler = scaler

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
        # add a normalization layer here
        inputs = self.scaler.transform(inputs)

        for w, b in params[:-1]:
            inputs = jnp.dot(w, inputs) + b
            inputs = self.activation(inputs) 
        w_f, b_f = params[-1]
        out = jnp.dot(w_f, inputs) + b_f
        return out
    
    def apply(self, params, x):
        f_pass_v = vmap(self.forward, in_axes=(None, 0))
        return f_pass_v(params, x)

    def full_Jacobian(self, params, x):
        def adjoint(params, x):
            jac = jax.jacfwd(self.forward, argnums=1)(params, x)
            return jac
        return vmap(adjoint, in_axes=(None, 0), out_axes=0)(params, x)
    
    def nn_adjoint(self, params, x):
        '''Calculate the Jacobian-vector product
            It is more efficient to do so. In this case, we use a vector to sum up the rows of the calculated
            Jacobian as the average model output gradient w.r.t. the input paramters.
        '''
        def adjoint_vect_prod(params, x):
            evaluated_value, vect_prod_fn = jax.vjp(self.forward, params, x)
            vect_prod = vect_prod_fn(jnp.ones(self.out_dim))[1] # shape same as the input dimension
            return vect_prod
        return vmap(adjoint_vect_prod, in_axes=(None, 0), out_axes=0)(params, x)

class Trainer:
    def __init__(self, net, num_epochs, batch_size, learning_rate, optimizer,  if_full_Jacobian='False'):
        '''Trainer for the adjoint matching neural network
        args:
            net: MLP class instance.
            num_epochs: maximum number of epochs for traning.
            batch_size: batch size for the training set.
            learning_rate: the initial learning rate for the optimizer.
            optimizer: optax optimizer.
            if_full_Jacobian: if matching the full Jacobian.
        '''
        self.net = net
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.if_full_Jacobian = if_full_Jacobian


    def loss(self, params, x, y, adj_y, alpha):
        # calcuate vector-true jacobian product
        def vect_jcob_prod(jcob):
            return jnp.ones(y.shape[1]) @ jcob # change the dimension to obtain the proper product
        true_v_j_prod = vmap(vect_jcob_prod, in_axes=0, out_axes=0)(adj_y)
        pred = self.net.apply(params, x)
        adj = self.net.nn_adjoint(params, x)
        adj_loss = jnp.mean((adj - true_v_j_prod)**2)
        totLoss = jnp.mean((pred - y)**2) + alpha*adj_loss
        return totLoss, adj_loss
    
    def loss_full_Jacobian(self, params, x, y, adj_y, alpha):
        pred = self.net.apply(params, x)
        adj = self.net.full_Jacobian(params, x)
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

    @partial(jax.jit, static_argnums=(0,)) 
    def step_full_Jacobian(self, params, x, y, adj_y, alpha, opt_state):
        (ls, adj_loss), grads = jax.value_and_grad(self.loss_full_Jacobian, argnums=0, has_aux=True)(params, x, y, adj_y, alpha)
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
            if (ep+1) % 10 == 0 and running_alpha <= alpha:
                running_alpha += 1 #alpha/10.
            for i in range(len(x_train)//self.batch_size):
                x_batch = x_train[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = y_train[i*self.batch_size:(i+1)*self.batch_size]
                adj_batch = adj_train[i*self.batch_size:(i+1)*self.batch_size]
                if self.if_full_Jacobian=='True':
                    params, ls, opt_state, adj_ls_batch= self.step_full_Jacobian(params, x_batch, y_batch,  adj_batch, running_alpha, opt_state)
                else:
                    params, ls, opt_state, adj_ls_batch= self.step_(params, x_batch, y_batch,  adj_batch, running_alpha, opt_state)
                pred_batch = self.net.apply(params, x_batch)
                train_running_r2.append(r2(y_batch, pred_batch))
                train_running_ls.append(ls)
                train_running_adj_ls.append(adj_ls_batch)

            if self.if_full_Jacobian=='True':
                if ep==0:
                    print("Training with full Jacobian...")
                ls_val, val_adj_loss = self.loss_full_Jacobian(params, x_val, y_val, adj_val, alpha) # set alpha with the preset value in validation. 
            else:
                if ep==0:
                    print("Training with vector-Jacobian product...")
                ls_val, val_adj_loss = self.loss(params, x_val, y_val, adj_val, alpha) # set alpha with the preset value in validation. 
            pred_val = self.net.apply(params, x_val)
            logger['train_loss'].append(jnp.asarray(train_running_ls).mean())
            logger['train_r2'].append(jnp.asarray(train_running_r2).mean())
            logger['train_adj_loss'].append(jnp.asarray(train_running_adj_ls).mean())
            logger['val_loss'].append(ls_val)
            logger['val_adj_loss'].append(val_adj_loss)
            logger['val_r2'].append(r2(y_val, pred_val))

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

            if ep + 1 == self.num_epochs:
                # save the params at the final epoch.
                logger['final_params'] = params

            with open('./logs/logger_'+save_date, 'wb') as f:
                pickle.dump(logger, f)
            if tol > patience:
                print("Early stopping...")
                break       
        return params    
    
    def predict(self, params, x):
        pred = self.net.apply(params, x)
        return pred



if __name__ == "__main__":
    from utils.data_loader import split_data, combine_burgers_data
    from utils.scaler import StandardScaler
    import os
    import argparse

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='AdjointMatchingNN')
    parser.add_argument('-a', type=float, default=1)
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-problem', type=str, default='Burgers')
    args = parser.parse_args()

    now = datetime.now()
    now_str = now.strftime('%m-%d-%H') + '_' + args.name + '_lr' + str(args.lr) +'_alpha' + str(args.a) 

    if args.problem == 'Glacier':
        data = np.load('../data/vary_beta.dat_4-7-23.npz')
        inputs = data['inputs']
        uout = data['uout']
        j_beta = data['jac_beta']
        jrav = data['jac_u']
        jrav = np.concatenate([jrav, j_beta[..., np.newaxis]], axis=-1)
        train, val, test = split_data(inputs, uout, jrav, shuffle_all=True)
    elif args.problem == 'Burgers':
        x, y, adj = combine_burgers_data('./Data/mixed_nu/')
        train, val, test = split_data(x, y, adj, shuffle_all=True)
    
    scaler = StandardScaler(train['x']) 
 
    net = MLP([50]*10, in_dim=train['x'].shape[1], out_dim=train['y'].shape[1], act_fn='relu', scaler=scaler)
    sup = Trainer(net=net, 
                num_epochs=args.epoch, 
                batch_size=args.batch_size, 
                learning_rate=args.lr, 
                optimizer=optax.adam,
                if_full_Jacobian='False')
    net_params = sup.train_(net.params, train, val, args.a, now_str)
    
    