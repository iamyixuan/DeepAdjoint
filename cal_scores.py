import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def plot_trend(mean, se, ylabel, min, max):
    '''plot the mean and standard error'''
    fig, ax = plt.subplots()
    var_names = ['LayerThickness', 'Salinity', 'Temperature', 'ZonalVelocity', 'MeridionalVelociy']
    x = np.arange(mean.shape[-1])
    for i in range(mean.shape[-2]):
        ax.plot(x, mean[i], label=var_names[i])
        ax.fill_between(x, mean[i] - se[i], mean[i] + se[i], alpha=.2)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(ylabel)
    ax.set_ylim(min, max)
    ax.set_box_aspect(1/1.62)
    ax.legend()
    return fig
    
    

# read predictions
if __name__ == "__main__":
    ROLLOUT = 0
    PLOT = 1
    if ROLLOUT == 1:
        with open('/pscratch/sd/y/yixuans/FNO-5var-rollout-trNoise.pkl', 'rb') as f:
            data = pickle.load(f)

        with open('./tmp/SOMA_mask.pkl', 'rb') as f:
            mask = pickle.load(f)
            mask = np.logical_or(mask['mask1'], mask['mask2'])


        overall_scores = [] # expected shape [10, 2, 5, 29] r2 and mape 
        for k in tqdm(range(len(data['true']))):
            true = data['true'][k].squeeze()
            pred = data['pred'][k].squeeze()


            true = np.transpose(true, axes=(0, 2, 3, 4, 1))
            pred = np.transpose(pred, axes=(0, 2, 3, 4, 1))[...,:-1]
            mask_b = mask[0:1,0:1,:,:,0:1]
            mask_b = np.broadcast_to(mask_b, true.shape)

            true[mask_b] = np.nan
            pred[mask_b] = np.nan

            vmin = [np.nanmin(true[...,i]) for i in range(true.shape[-1])]
            vmax = [np.nanmax(true[...,i]) for i in range(true.shape[-1])]
            var_names = ['LayerThickness', 'Salinity', 'Temperature', 'ZonalVelocity', 'MeridionalVelociy']

            r2_timestep = []
            mape_timestep = []
            for i in range(true.shape[-1]):
                for t in range(29):
                    cur_true = np.nan_to_num(true[t,...,i])
                    cur_pred = np.nan_to_num(pred[t,...,i])
                    r2_timestep.append(r2(cur_true, cur_pred))
                    mape_timestep.append(sMAPE(cur_true, cur_pred))
                    fig_true = plot_field(true[t,10,..., i], pred[t,10,..., i], vmin[i], vmax[i])

                
                # fig_pred = plot_field(pred[15,10, ..., i], vmin, vmax)

                    fig_true.savefig(f'./eval_plots/rollout/FNO/rollout-with-train-noise/true-pred-UNet-var{var_names[i]}-t{t}.png', format='png', dpi=200)
                    plt.close()
                # fig_pred.savefig('eval_plots/pred-ResNet-5-var.png', format='png', dpi=200)
            scores = np.stack([np.array(r2_timestep).reshape(5, 29), np.array(mape_timestep).reshape(5, 29)])
            overall_scores.append(scores)

        # metric_true = np.nan_to_num(true)
        # metric_pred = np.nan_to_num(pred)

        # r2_scores = apply_to_channel(metric_true, metric_pred, r2)
        # sMAPE_scores = apply_to_channel(metric_true, metric_pred, sMAPE)
        # rMSE_scores = apply_to_channel(metric_true, metric_pred, rMSE)

        # print(r'$R^2$ scores are', format_vals(r2_scores))
        # print(r'$sMAPE$ scores are', format_vals(sMAPE_scores))
        # print(r'$rMSE$ scores are', format_vals(rMSE_scores))
        # overall_scores = np.array(overall_scores)

        np.save('./eval_plots/overall_scores-FNO-trNoise.npy', overall_scores)
    
    if PLOT == 1:
        data = np.load('./eval_plots/overall_scores-FNO-trNoise.npy')
        
        from scipy.stats import sem
        mu = np.mean(data, axis=0)
        se = sem(data, axis=0)
        fig1 = plot_trend(mu[0], se[0], r'$R^2$', min=-0.1, max=1.1)
        fig1.savefig('./eval_plots/r2_rollout-FNP-trNoise.png', format='png', dpi=300)
        fig2 = plot_trend(mu[1], se[1], r'$sMAPE$', min=0, max=100)
        fig2.savefig('./eval_plots/mape_rollout-FNO-trNoise.png', format='png', dpi=300)
        