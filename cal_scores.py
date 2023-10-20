import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
# import metrics
from deep_adjoint.utils.metrics import r2, rMSE, sMAPE

# plot utilies

def plot_field(true, pred, vmin, vmax):
    '''field: shape [x, y]'''
    data = [true, pred]
    fig, axs = plt.subplots(1,2, figsize=(10,8))
    for i, ax in enumerate(axs.ravel()):
        im = ax.imshow(data[i].T, 
                cmap='seismic',
                origin='lower',
                aspect='auto',
                vmin=vmin,
                vmax=vmax)
        ax.set_box_aspect(1)
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=20)
        
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


def plot_trend(mean, se, ylabel, min, max, var_idx):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 20
    '''plot the mean and standard error'''
    fig, ax = plt.subplots()
    # var_names = ['LayerThickness', 'Salinity', 'Temperature', 'ZonalVelocity', 'MeridionalVelociy']
    m_names = ['ResNet', 'U-Net', 'FNO']
    x = np.arange(mean.shape[-1])
    for i in range(mean.shape[0]):
        ax.plot(x, mean[i, var_idx], label=m_names[i])
        ax.fill_between(x, mean[i, var_idx] - se[i,var_idx], mean[i, var_idx] + se[i,var_idx], alpha=.2)
    ax.set_xlabel('Time Steps', fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_ylim(min, max)
    ax.set_box_aspect(1/1.62)
    ax.grid(which='major',  color='lightgray', linestyle='-.')
    ax.legend()
    return fig
    
    

# read predictions
if __name__ == "__main__":
    ROLLOUT = 1
    PLOT = 0
    CAL_SCORE = 0
    PLOT_SINGLE = 1
    if ROLLOUT == 1:
        with open('/pscratch/sd/y/yixuans/2023-10-2-FNO-redi-predictions.pkl', 'rb') as f:
            data = pickle.load(f)
    
        with open('./tmp/SOMA_mask.pkl', 'rb') as f:
            mask = pickle.load(f)
            mask = np.logical_or(mask['mask1'], mask['mask2'])

        DIR_NAME = 'FNO-redi'
        if not os.path.exists(f'./eval_plots/{DIR_NAME}/'):
            os.makedirs(f'./eval_plots/{DIR_NAME}/')
        overall_scores = [] # expected shape [10, 2, 5, 29] r2 and mape 
        for k in tqdm(range(len(data['true']))):
            true = data['true'][k].squeeze()
            pred = data['pred'][k].squeeze()

            true = np.transpose(true, axes=(0, 2, 3, 4, 1))
            pred = np.transpose(pred, axes=(0, 2, 3, 4, 1))# [...,:-1] # enable this while rollout
            mask_b = mask[0:1,0:1,:,:,0:1]
            mask_b = np.broadcast_to(mask_b, true.shape)
           
            true[mask_b] = np.nan
            pred[mask_b] = np.nan

            vmin = [np.nanmin(true[...,i]) for i in range(true.shape[-1])]
            vmax = [np.nanmax(true[...,i]) for i in range(true.shape[-1])]
            var_names = ['LayerThickness', 'Salinity', 'Temperature', 'ZonalVelocity', 'MeridionalVelociy']
            
            if PLOT_SINGLE == 1:
                for v in range(len(var_names)):
                    fig = plot_field(true[7, 10, ..., v], pred[7, 10, ..., v], vmin[v], vmax[v])
                    fig.savefig(f'./eval_plots/{DIR_NAME}/var{var_names[v]}-onestep.pdf', format='pdf', bbox_inches='tight')
                break

            r2_timestep = []
            mape_timestep = []
            for i in range(true.shape[-1]):
                for t in range(29):
                    if t != 15:
                        pass
                    else:
                        cur_true = np.nan_to_num(true[t,...,i])
                        cur_pred = np.nan_to_num(pred[t,...,i])
                        r2_timestep.append(r2(cur_true, cur_pred))
                        mape_timestep.append(sMAPE(cur_true, cur_pred))
                        fig_true = plot_field(true[t,10,..., i], pred[t,10,..., i], vmin[i], vmax[i])

                
                # fig_pred = plot_field(pred[15,10, ..., i], vmin, vmax)
                        
                        fig_true.savefig(f'./eval_plots/{DIR_NAME}/var{var_names[i]}-t{t}.pdf', format='pdf', bbox_inches='tight')
                        plt.close()
                    
                # fig_pred.savefig('eval_plots/pred-ResNet-5-var.png', format='png', dpi=200)
            # scores = np.stack([np.array(r2_timestep).reshape(5, 29), np.array(mape_timestep).reshape(5, 29)])
            # overall_scores.append(scores)

        # metric_true = np.nan_to_num(true)
        # metric_pred = np.nan_to_num(pred)

        # r2_scores = apply_to_channel(metric_true, metric_pred, r2)
        # sMAPE_scores = apply_to_channel(metric_true, metric_pred, sMAPE)
        # rMSE_scores = apply_to_channel(metric_true, metric_pred, rMSE)

        # print(r'$R^2$ scores are', format_vals(r2_scores))
        # print(r'$sMAPE$ scores are', format_vals(sMAPE_scores))
        # print(r'$rMSE$ scores are', format_vals(rMSE_scores))
        # overall_scores = np.array(overall_scores)

        # np.save('./eval_plots/overall_scores-FNO-trNoise.npy', overall_scores)
    if CAL_SCORE == 1: 
        with open('/pscratch/sd/y/yixuans/2023-10-2-FNO-cvmix-predictions.pkl', 'rb') as f:
            data = pickle.load(f)

        with open('./tmp/SOMA_mask.pkl', 'rb') as f:
            mask = pickle.load(f)
            mask = np.logical_or(mask['mask1'], mask['mask2'])
        print(mask.shape)
        true = np.concatenate(data['true'])
        pred = np.concatenate(data['pred'])
        true = np.transpose(true, axes=(0, 2, 3, 4, 1))
        pred = np.transpose(pred, axes=(0, 2, 3, 4, 1))
        print(true.shape, pred.shape)
        mask_b = mask[0:1,0:1,:,:,0:1]
        mask_b = np.broadcast_to(mask_b, true.shape)
        
        true[mask_b] = np.nan
        pred[mask_b] = np.nan

        metric_true = np.nan_to_num(true)
        metric_pred = np.nan_to_num(pred)

        r2_scores = apply_to_channel(metric_true, metric_pred, r2)
        sMAPE_scores = apply_to_channel(metric_true, metric_pred, sMAPE)
        rMSE_scores = apply_to_channel(metric_true, metric_pred, rMSE)

        print(r'$R^2$ scores are', format_vals(r2_scores))
        print(r'$sMAPE$ scores are', format_vals(sMAPE_scores))
        print(r'$rMSE$ scores are', format_vals(rMSE_scores))
    
    if PLOT == 1:
        data_Res = np.load('./eval_plots/overall_scores-ResNet.npy')
        data_Unet = np.load('./eval_plots/overall_scores-UNet.npy')
        data_FNO = np.load('./eval_plots/overall_scores-FNO.npy')
        data = np.array([data_Res, data_Unet, data_FNO])
        print(data_Res.shape)
        var_names = ['LayerThickness', 'Salinity', 'Temperature', 'ZonalVelocity', 'MeridionalVelociy'] 
        from scipy.stats import sem
        mu = np.mean(data, axis=1)
        se = sem(data, axis=1)

        var_idx = 2
        fig1 = plot_trend(mu[:,0], se[:,0], r'$R^2$', min=-0.1, max=1.1, var_idx=var_idx)
        fig1.savefig(f'./eval_plots/r2_{var_names[var_idx]}-zoom.pdf', format='pdf', bbox_inches='tight')
        fig2 = plot_trend(mu[:, 1], se[:,1], r'$sMAPE$', min=0, max=100, var_idx=var_idx)
        fig2.savefig(f'./eval_plots/mape_rollout_{var_names[var_idx]}-zoom.pdf', format='pdf', bbox_inches='tight')
        