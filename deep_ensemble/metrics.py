import numpy as np
import torch
import torch.nn as nn


def r2(true, pred):
    if np.isnan(true).any():
        true = true.reshape(
            -1,
        )
        pred = pred.reshape(
            -1,
        )
        mask = np.isnan(true)
        true = true[~mask]
        pred = pred[~mask]
    ss_res = np.sum(np.power(true - pred, 2), axis=(0, 2, 3))
    ss_tot = np.sum(
        np.power(true - np.mean(true, axis=(0, 2, 3), keepdims=True), 2),
        axis=(0, 2, 3),
    )
    return 1 - ss_res / ss_tot


def anomalyCorrelationCoef(true, pred):
    if np.isnan(true).any():
        true = true.reshape(
            -1,
        )
        pred = pred.reshape(
            -1,
        )
        mask = np.isnan(true)
        true = true[~mask]
        pred = pred[~mask]

    trueMean = np.mean(true, axis=(0, 2, 3), keepdims=True)
    trueAnomaly = true - trueMean
    predAnomaly = pred - trueMean

    cov = np.mean(predAnomaly * trueAnomaly, axis=(0, 2, 3))
    std = np.sqrt(
        np.mean(predAnomaly**2, axis=(0, 2, 3))
        * np.mean(trueAnomaly**2, axis=(0, 2, 3))
    )
    return cov / std


class QuantileLoss(nn.Module):
    def __init__(self):
        super(QuantileLoss, self).__init__()
        self.quantiles = [0.33, 0.5, 0.66]

    def forward(self, true, pred):
        """
        true and pred have shape (B, ch*3, 100, 100)
        we should calculate the channel wise scores
        """
        loss = 0
        ch = true.shape[1]
        for i, q in enumerate(self.quantiles):
            pred_qauntile = pred[:, i * ch : (i + 1) * ch, :, :]
            loss += torch.mean(
                torch.max(
                    q * (true - pred_qauntile),
                    (q - 1) * (true - pred_qauntile),
                )
            )
        return loss


class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, true, pred):
        """
        mean and std have shape (B, 4, 100, 100)
        true has shape (B, 4, 100, 100)

        """
        ch = true.shape[1]
        mean = pred[:, :ch, :, :]
        var = pred[:, ch:, :, :]
        a = 0.5 * torch.log(var)
        b = 0.5 * ((true - mean) ** 2 / var)
        # print(var.mean().item())
        # return torch.mean(
        #     0.5 * torch.log(2 * np.pi * var)
        #     + 0.5 * ((true - mean)**2 / var)
        # )
        return torch.mean(a + b)


class ACCLoss(nn.Module):
    def __init__(self):
        super(ACCLoss, self).__init__()

    def forward(self, true, pred):
        """
        true and pred have shape (B, 5, 100, 100)
        we should calculate the channel wise scores
        """
        TrueMean = torch.mean(true, dim=(0, 2, 3), keepdims=True)
        TrueAnomaly = true - TrueMean
        PredAnomaly = pred - TrueMean

        cov = torch.mean(PredAnomaly * TrueAnomaly, dim=(0, 2, 3))
        std = torch.sqrt(
            torch.mean(PredAnomaly**2, dim=(0, 2, 3))
            * torch.mean(TrueAnomaly**2, dim=(0, 2, 3))
        )
        return -torch.mean(cov / std)


class MSE_ACCLoss(nn.Module):
    def __init__(self, alpha):
        super(MSE_ACCLoss, self).__init__()
        self.alpha = alpha

    def forward(self, true, pred):
        """
        true and pred have shape (B, 5, 100, 100)
        we should calculate the channel wise scores
        """
        TrueMean = torch.mean(true, dim=(0, 2, 3), keepdims=True)
        TrueAnomaly = true - TrueMean
        PredAnomaly = pred - TrueMean

        cov = torch.mean(PredAnomaly * TrueAnomaly, dim=(0, 2, 3))
        std = torch.sqrt(
            torch.mean(PredAnomaly**2, dim=(0, 2, 3))
            * torch.mean(TrueAnomaly**2, dim=(0, 2, 3))
        )

        acc_term = -torch.mean(cov / std)
        mse_term = torch.mean((true - pred) ** 2)

        return self.alpha * mse_term + (1 - self.alpha) * acc_term
