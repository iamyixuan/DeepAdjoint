import numpy as np
import pickle
import matplotlib.pyplot as plt

# import metrics
from deep_adjoint.utils.metrics import r2, rMSE, sMAPE

# plot utilies

def plot_field(true, pred, vmin, vmax):
    '''field: shape [x, y]'''
    data = [true, pred]
    fig, axs = plt.subplots(1,2, figsize=(10,8))
    for i, ax in enumerate(axs.ravel()):
        im = ax.imshow(data[i].T, 
                cmap='rainbow',
                origin='lower',
                aspect='auto',
                vmin=vmin,
                vmax=vmax)
        ax.set_box_aspect(1)
        # ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    return fig

def apply_to_channel(true, pred, metric_fn):
    scores = []
    print(true.shape)
    print(f'number of channels is {true.shape[-1]}')
    for i in range(true.shape[-1]):
        scores.append(metric_fn(true[..., i], pred[..., i]))
    return scores

def format_vals(scores):
    return [f"{s:.3e}" for s in scores]

# read predictions

with open('/pscratch/sd/y/yixuans/2023-8-30-5-var-pred-Unet.pkl', 'rb') as f:
    data = pickle.load(f)

with open('./tmp/SOMA_mask.pkl', 'rb') as f:
    mask = pickle.load(f)

mask = np.logical_or(mask['mask1'], mask['mask2'])
true = data['true'][0]
pred = data['pred'][0]

true = np.transpose(true, axes=(0, 2, 3, 4, 1))
pred = np.transpose(pred, axes=(0, 2, 3, 4, 1))
mask_b = mask[0:1,0:1,:,:,0:1]
mask_b = np.broadcast_to(mask_b, true.shape)

true[mask_b] = np.nan
pred[mask_b] = np.nan

vmin = [np.nanmin(true[...,i]) for i in range(true.shape[-1])]
vmax = [np.nanmax(true[...,i]) for i in range(true.shape[-1])]
var_names = ['LayerThickness', 'Salinity', 'Temperature', 'ZonalVelocity', 'MeridionalVelociy']
for i in range(true.shape[-1]):
    fig_true = plot_field(true[15,10,..., i], pred[15,10,..., i], vmin[i], vmax[i])
    # fig_pred = plot_field(pred[15,10, ..., i], vmin, vmax)

    fig_true.savefig(f'eval_plots/pred-true-UNet/true-pred-UNet-var{var_names[i]}-5input.png', format='png', dpi=200)
    # fig_pred.savefig('eval_plots/pred-ResNet-5-var.png', format='png', dpi=200)

metric_true = np.nan_to_num(true)
metric_pred = np.nan_to_num(pred)

r2_scores = apply_to_channel(metric_true, metric_pred, r2)
sMAPE_scores = apply_to_channel(metric_true, metric_pred, sMAPE)
rMSE_scores = apply_to_channel(metric_true, metric_pred, rMSE)

print(r'$R^2$ scores are', format_vals(r2_scores))
print(r'$sMAPE$ scores are', format_vals(sMAPE_scores))
print(r'$rMSE$ scores are', format_vals(rMSE_scores))