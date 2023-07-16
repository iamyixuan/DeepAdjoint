from VJPMatching import MLP, Trainer
from deep_adjoint.utils.data import split_data, combine_burgers_data
from deep_adjoint.utils.metrics import mean_squared_error, r2
from deep_adjoint.utils.scaler import StandardScaler
import pickle
import optax
import numpy as np
from jax import vmap

# config.update("jax_enable_x64", True)

# data = np.load('../data/vary_beta.dat_4-7-23.npz')
# inputs = data['inputs']
# uout = data['uout']
# j_beta = data['jac_beta']
# jrav = data['jac_u']
# jrav = np.concatenate([jrav, j_beta[..., np.newaxis]], axis=-1)

def vect_jcob_prod(v, jcob):
            return (
                v @ jcob
            )  # change the dimension to obtain the proper product

        

true_vjp = vmap(vect_jcob_prod, in_axes=0, out_axes=0)




x, y, adj = combine_burgers_data("./deep_adjoint/Data/mixed_nu/")
print(adj.shape)
print(np.linalg.norm(adj[..., -1]))
print(np.max(adj[...,-1]))
print(np.min(adj[..., -1]))
print(np.mean(adj[...,-1]))

train, val, test = split_data(x, y, adj, shuffle_all=True)

scaler = StandardScaler(train["x"])

net = MLP(
    [200] * 10,
    in_dim=train["x"].shape[1],
    out_dim=train["y"].shape[1],
    act_fn="tanh",
    scaler=scaler,
)
# sup = Trainer(
#     net=net,
#     num_epochs=args.epoch,
#     batch_size=args.batch_size,
#     learning_rate=args.lr,
#     optimizer=optax.adam,
#     if_full_Jacobian="False",
# )

with open('./logs/logger_07-13-00_AdjointMatchingNN_lr0.0001_alpha1', 'rb') as f:
    logger = pickle.load(f)


# train, val, test = split_data(x, y, adj, shuffle_all=True)
# scaler = StandardScaler(train['x'])
x = test['x'][:2000]
# net = MLP([200]*5, in_dim=train['x'].shape[1], out_dim=train['y'].shape[1], act_fn='tanh', scaler=scaler)

params = logger['final_params']
test_pred = net.apply(params, x)


# test_pred = y_scaler.inverse_transform(test_pred)
v = test_pred - test['y'][:2000]

print(x.shape, v.shape)

pred_adj_p = net.full_Jacobian(params, x)[:, -1]
pred_adj = net.nn_adjoint(params, x, v)
# print(pred_adj.shape)
# true_adj = true_vjp(v, test['adj'][:2000])
# # pred_adj = adj_scaler.inverse_transform(pred_adj)

# # u_pred = net.apply(params, x)
# # u_pred = y_scaler.inverse_transform(u_pred)


print('The test MSE is {:.4f}'.format(mean_squared_error(test['y'][:2000], test_pred)))
# print('The train MSE is {:.4f}'.format(mean_squared_error(train['y'], u_pred)))
print('The test R2 is {:.4f}'.format(r2(test['y'][:2000], test_pred)))
# print('The test adj mse is {:.4f}'.format(mean_squared_error(true_adj, pred_adj)))
# print('The test adj R2 is {:4f}'.format(r2(true_adj, pred_adj)))

print('the jac wrt the parameter R2 is {:.4f}'.format(r2(test['adj'][:2000][:, -1], pred_adj_p)))
print('The jac parameter only Mse is {}'.format(mean_squared_error(test['adj'][:2000][:, -1], pred_adj_p)))