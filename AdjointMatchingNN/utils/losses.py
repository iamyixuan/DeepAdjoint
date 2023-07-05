import torch 
import numpy as np

class Losses:
    def __init__(self, loss_name) -> None:
        '''
        loss: [MSE, MAE]
        '''
        self.loss_name = loss_name
    
    def __call__(self):
        ls_fn = self.get_loss(self.loss_name)
        return ls_fn
    
    def get_lam(self, true, pred, adj_true):
        '''
        true and pred have the shape of [num_data, timesteps, x_locs]
        adj_true has the shape of [num_data, timesteps, y_locs, x_locs]
        '''
        init_lam = 2 * (true[:, -1, :] - pred[:, -1, :]) # shape [num_data, x_locs, 1]
        lam = [init_lam]
        for t in range(1, true.shape[-1]):
            lam_T_t = 2 * (true[:, -1-t, :] - pred[:, -1-t, :]) + lam[-1] @ adj_true[:, -1-t, :, :]
            lam.append(lam_T_t)
        # the lamdbas are from the last timestep to the first, so we need to reverse it to return
        return np.array(lam.reverse())


    def MSE(self, true, pred):
        loss = torch.pow(true - pred, 2)
        return torch.mean(loss)

    def LagrangianLoss(self, true, pred, adj):
        # implement the Lagrangian of the optimization problem.
        lam = self.get_lam(true, pred, adj)
        L = self.MSE(true, pred) + lam * (true - pred)
        return L 
    

    def MAE(self, true ,pred):
        loss = torch.abs(true - pred)
        return torch.mean(loss)

    def get_loss(self, loss_name):
        if loss_name == "MSE":
            return self.MSE
        elif loss_name == "MAE":
            return self.MAE
        else:
            raise Exception('Loss name not recognized!')
        