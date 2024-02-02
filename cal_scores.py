import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import h5py
# import metrics
from deep_adjoint.utils.metrics import r2, NRMSE, anomalyCorrelationCoef

# plot utilies
def rescale(t, p, idx):
    data = h5py.File('/pscratch/sd/y/yixuans/datatset/SOMA/thedataset-GM-dayAvg-2.hdf5', 'r')['forward_0'][..., [7, 8, 11, 14, 15]]
    mask = np.logical_or(data>1e16, data<-1e16)
    data[mask] = np.nan
    t[mask[0,0, :, :,0]] = np.nan
    p[mask[0,0, :, :,0]] = np.nan
    MIN = np.nanmin(data, axis=(0, 1, 2, 3))[idx]
    MAX = np.nanmax(data, axis=(0, 1, 2, 3))[idx]
    print(f"Minimum value {MIN}, Maximum value {MAX}")

    t_rescale = t * (MAX - MIN) + MIN
    p_rescale = p * (MAX - MIN) + MIN 
    return t_rescale, p_rescale




def plot_field(true, pred, cbarlabel):
    '''field: shape [x, y]'''
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2

    vmin = np.min([np.nanmin(true), 0])
    vmax =  np.nanmax(true)
    # print(vmin, vmax)
    data = [true, pred, true - pred]

    fig, axs = plt.subplots(3, 1, figsize=(4, 12))
    for i, ax in enumerate(axs.ravel()):
        if i == 2:
            im2 = ax.imshow(data[i].T, 
                    cmap='turbo',
                    origin='lower',
                    aspect='auto',
                    vmin=-2,
                    vmax=2)
        else:
            im = ax.imshow(data[i].T, 
                        cmap='turbo',
                        origin='lower',
                        aspect='auto',
                        vmin=vmin,
                        vmax=vmax)
        if i == 0:
            pos1 = ax.get_position()
        if i == 1:
            pos2 = ax.get_position()
        if i == 2:
            pos3 = ax.get_position()
    
        ax.set_box_aspect(1)
        ax.axis('off')
        # cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cb.ax.tick_params(labelsize=20)

    cbar_ax = fig.add_axes([0.9, pos2.bounds[1], 0.05, pos1.bounds[1] + pos1.bounds[-1] - pos2.bounds[1]])
    cbar_ax2 = fig.add_axes([.9, pos3.bounds[1], 0.05, pos3.bounds[3] ])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='vertical')
    fig.suptitle(f'{cbarlabel}', fontsize=20, y=.91)
    # cbar.set_label(f'{cbarlabel}', fontsize=16) 
    # plt.tight_layout()
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


def plot_trend(mean, se, ylabel, min, max, var_idx, m_names):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2
    '''plot the mean and standard error'''
    fig, ax = plt.subplots()
    # var_names = ['LayerThickness', 'Salinity', 'Temperature', 'ZonalVelocity', 'MeridionalVelociy']
    #m_names = ['ResNet', 'U-Net', 'FNO']
    x = np.arange(mean.shape[-1])
    for i in range(mean.shape[0]):
        ax.plot(x, mean[i, var_idx], label=m_names[i])
        ax.fill_between(x, mean[i, var_idx] - se[i,var_idx], mean[i, var_idx] + se[i,var_idx], alpha=.2)
    ax.set_xlabel('Time Steps', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_ylim(min, max)
    ax.set_box_aspect(1/1.62)
    ax.grid(True, which='both',  color='lightgray', linestyle='-.', linewidth=1)
    ax.legend()
    return fig
    
    

# read predictions
if __name__ == "__main__":
    ROLLOUT = 0
    PLOT = 1
    CAL_SCORE = 0
    PLOT_SINGLE = 0
    DATA = 'GM_D_AVG'
    NET = 'FNO'

    var_names = [ 'Salinity', 'Temperature', 'Layer Thickness', 'Meridional Velocity', 'Zonal Velocity']
    with open(f'/pscratch/sd/y/yixuans/FNO-{DATA}-predictions.pkl', 'rb') as f:
        data = pickle.load(f)
    true = data['true'][1]
    pred = data['pred'][1]

    true = np.transpose(true, axes=(0, 2, 3, 4, 1))
    pred = np.transpose(pred, axes=(0, 2, 3, 4, 1))
    varID = 4
    t = true[2, 10,..., varID]
    p = pred[2, 10, ..., varID]

    t, p = rescale(t, p, varID)
    
    fig = plot_field(t, p, var_names[varID])
    fig.savefig(f'./eval_plots/{DATA}-truePred{var_names[varID]}.pdf', format='pdf', bbox_inches='tight')
    

        
    if ROLLOUT == 1:
        # with open(f'/pscratch/sd/y/yixuans/FNO-{DATA}-rollout.pkl', 'rb') as f:
        #     data = pickle.load(f)

        with open(f'/pscratch/sd/y/yixuans/{NET}-{DATA}-rollout.pkl', 'rb') as f:
            data = pickle.load(f)
    
        with open('./tmp/SOMA_mask.pkl', 'rb') as f:
            mask = pickle.load(f)
            mask = np.logical_or(mask['mask1'], mask['mask2'])

        DIR_NAME = f'{NET}-{DATA}'
        if not os.path.exists(f'./eval_plots/{DIR_NAME}/'):
            os.makedirs(f'./eval_plots/{DIR_NAME}/')
        overall_scores = [] # expected shape [10, 2, 5, 29] r2 and mape 
        for k in tqdm(range(len(data['true']))):
            true = data['true'][k].squeeze()
            pred = data['pred'][k].squeeze()

            true = np.transpose(true, axes=(0, 2, 3, 4, 1))
            pred = np.transpose(pred, axes=(0, 2, 3, 4, 1))[...,:-1] # enable this while rollout
            mask_b = mask[0:1,0:1,:,:,0:1]
            mask_b = np.broadcast_to(mask_b, true.shape)
           
            true[mask_b] = np.nan
            pred[mask_b] = np.nan

            vmin = [np.nanmin(true[...,i]) for i in range(true.shape[-1])]
            vmax = [np.nanmax(true[...,i]) for i in range(true.shape[-1])]
            #var_names = ['LayerThickness', 'Salinity', 'Temperature', 'ZonalVelocity', 'MeridionalVelociy']
            var_names = ['Salinity', 'Temperature', 'Layer Thickness', 'Meridional Velocity', 'Zonal Velocity']
            
            if PLOT_SINGLE == 1:
                for v in range(len(var_names)):
                    fig = plot_field(true[7, 10, ..., v], pred[7, 10, ..., v], vmin[v], vmax[v])
                    fig.savefig(f'./eval_plots/{DIR_NAME}/var{var_names[v]}-onestep.png', format='png', bbox_inches='tight', dpi=300, transparent=True)
                break

            r2_timestep = []
            NCC_timestep = []
            NRMSE_timestep = []
            for i in range(true.shape[-1]):
                for t in range(29):
                    # if t != 15:
                    #     pass
                    # else:
                    cur_true = true[t, ...,i]
                    cur_pred = pred[t, ...,i]
                    r2_timestep.append(r2(cur_true, cur_pred))
                    NCC_timestep.append(anomalyCorrelationCoef(cur_true, cur_pred))
                    NRMSE_timestep.append(NRMSE(cur_true, cur_pred))

    
                    fig_true = plot_field(true[t,10,..., i], pred[t,10,..., i], var_names[i])
                    # print(NCC_timestep[-1])

            
            # fig_pred = plot_field(pred[15,10, ..., i], vmin, vmax)
               
                    fig_true.savefig(f'./eval_plots/{DIR_NAME}/var{var_names[i]}-t{t}.pdf', format='pdf', bbox_inches='tight')
                    plt.close()
                    
                # fig_pred.savefig('eval_plots/pred-ResNet-5-var.png', format='png', dpi=200)
            scores = np.stack([np.array(r2_timestep).reshape(5, 29), np.array(NRMSE_timestep).reshape(5, 29), np.array(NCC_timestep).reshape(5, 29)])
            overall_scores.append(scores)

        # metric_true = np.nan_to_num(true)
        # metric_pred = np.nan_to_num(pred)

        # r2_scores = apply_to_channel(metric_true, metric_pred, r2)
        # sMAPE_scores = apply_to_channel(metric_true, metric_pred, sMAPE)
        # rMSE_scores = apply_to_channel(metric_true, metric_pred, rMSE)

        # print(r'$R^2$ scores are', format_vals(r2_scores))
        # print(r'$sMAPE$ scores are', format_vals(sMAPE_scores))
        # print(r'$rMSE$ scores are', format_vals(rMSE_scores))
        overall_scores = np.array(overall_scores)

        np.save(f'./eval_plots/{DIR_NAME}/overall_scores-{DATA}-rollout.npy', overall_scores)
    if CAL_SCORE == 1: 
        # with open(f'/pscratch/sd/y/yixuans/FNO-{DATA}-predictions.pkl', 'rb') as f:
        #     data = pickle.load(f)
        
        with open(f'/pscratch/sd/y/yixuans/FNO-GM_D_AVG-predictions.pkl', 'rb') as f:
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

        metric_true = true #np.nan_to_num(true)
        metric_pred = pred #np.nan_to_num(pred)

        r2_scores = apply_to_channel(metric_true, metric_pred, r2)
        sMAPE_scores = apply_to_channel(metric_true, metric_pred, NRMSE)
        rMSE_scores = apply_to_channel(metric_true, metric_pred, anomalyCorrelationCoef)

        print(r'$R^2$ scores are', format_vals(r2_scores))
        print(r'$NRMSE$ scores are', format_vals(sMAPE_scores))
        print(r'$ACC$ scores are', format_vals(rMSE_scores))
    
    if PLOT == 1:
        # data_Res = np.load('/global/homes/y/yixuans/DeepAdjoint/eval_plots/ResNet-5var/overall_scores-5var-rollout.npy')
        # data_Unet = np.load('/global/homes/y/yixuans/DeepAdjoint/eval_plots/Unet-5var/overall_scores-5var-rollout.npy')
        data_FNO = np.load('/global/homes/y/yixuans/DeepAdjoint/eval_plots/FNO-GM_D_AVG/overall_scores-GM_D_AVG-rollout.npy')
        # data = np.array([data_Res, data_Unet, data_FNO])

        data_REDI = np.load('/global/homes/y/yixuans/DeepAdjoint/eval_plots/FNO-REDI/overall_scores-REDI-rollout.npy')
        data_CVMIX = np.load('/global/homes/y/yixuans/DeepAdjoint/eval_plots/FNO-CVMIX/overall_scores-CVMIX-rollout.npy')
        data_bottomDrag = np.load('/global/homes/y/yixuans/DeepAdjoint/eval_plots/FNO-BTTMDRAG/overall_scores-BTTMDRAG-rollout.npy')
        print(data_REDI.shape, data_CVMIX.shape, data_bottomDrag.shape)
        #data = [data_FNO, data_REDI, data_CVMIX, data_bottomDrag]
        data = [data_FNO]
        # print(data[2][1, 1,...])
        #var_names = ['LayerThickness', 'Salinity', 'Temperature', 'ZonalVelocity', 'MeridionalVelociy'] 
        var_names = ['Salinity', 'Temperature', 'Layer Thickness', 'Meridional Velocity', 'Zonal Velocity']
        from scipy.stats import sem
        # mu = np.mean(data, axis=1)
        # se = sem(data, axis=1)
        mu = np.array([np.mean(d, axis=0) for d in data])
        se = np.array([sem(d, axis=0) for d in data])
        print(mu.shape, se.shape)

        var_idx = 1
        m_names = [r'$\kappa_{GM}$', r'$\kappa_{redi}$', r'$\kappa_{bg}$', r'$C_D$'] 
        
        fig1 = plot_trend(mu[:,1], se[:,1], r'$NRMSE$', min=0, max=100, var_idx=var_idx, m_names=m_names)
        fig1.savefig(f'./eval_plots/0NRMSE_rollout_{var_names[var_idx]}-GM_D_AVG.pdf', format='pdf', bbox_inches='tight') 
        fig2 = plot_trend(mu[:, 2], se[:,2], r'$ACC$', min=0, max=1.1, var_idx=var_idx, m_names=m_names)
        fig2.savefig(f'./eval_plots/0ACC_rollout_{var_names[var_idx]}-GM_D_AVG.pdf', format='pdf', bbox_inches='tight')
        