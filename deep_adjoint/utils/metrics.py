# import jax.numpy as jnp
import numpy as np


# def mean_squared_error(true, pred):
#     return jnp.mean(jnp.power(true - pred, 2))
def get_metrics(true, pred):
    coef_det = r2(true, pred)
    mape = sMAPE(true, pred)
    rel_mse = rMSE(true, pred)
    return np.array([coef_det, mape, rel_mse])


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
    ss_res = np.sum(np.power(true - pred, 2))
    ss_tot = np.sum(np.power(true - np.mean(true), 2))
    return 1 - ss_res / ss_tot


# def mape(true, pred):
#     return jnp.abs(true - pred).mean() / jnp.abs(true).mean() * 100


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
    trueMean = np.mean(true)
    trueAnomaly = true - trueMean
    predAnomaly = pred - trueMean

    cov = np.mean(predAnomaly * trueAnomaly)
    std = np.sqrt(np.mean(predAnomaly**2) * np.mean(trueAnomaly**2))
    return cov / std


def normalizedCrossCorrelation(true, pred):
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

    if true.shape != pred.shape:
        raise ValueError("The input arrays must have the same shape.")

    # Subtract the mean of each array to make them zero-mean
    true_mean_subtracted = true - np.mean(true)
    pred_mean_subtracted = pred - np.mean(pred)

    # Compute the cross-correlation term
    numerator = np.sum(true_mean_subtracted * pred_mean_subtracted)

    # Compute the normalization terms
    denominator = np.sqrt(np.sum(true_mean_subtracted**2)) * np.sqrt(
        np.sum(pred_mean_subtracted**2)
    )

    # Compute the NCC
    ncc = numerator / denominator

    return ncc


def NRMSE(true, pred):
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
    max_ = np.max(true)
    min_ = np.min(true)
    squaredError = np.sqrt(np.power((true - pred), 2))
    return np.mean(squaredError / (max_ - min_)) * 100


def rMSE(true, pred):
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
    return (
        np.mean(
            np.power(true - pred, 2)
            / (0.5 * (np.power(true, 2) + np.power(pred, 2) + 1e-12))
        )
        * 100
    )


# def sMAPE(true, pred):
#     if np.isnan(true).any():
#         true = true.reshape(-1,)
#         pred = pred.reshape(-1,)
#         mask = np.isnan(true)
#         true = true[~mask]
#         pred = pred[~mask]
#     return np.nanmean(np.abs(true - pred) / (.5 * (np.abs(true) + np.abs(pred)))) * 100


class min_max_scaler:
    def __init__(self, d_min, d_max, s_min=0, s_max=100) -> None:
        self.d_min = d_min
        self.d_max = d_max
        self.s_min = s_min
        self.s_max = s_max

    def transform(self, x):
        d_diff = self.d_max - self.d_min
        s_diff = self.s_max - self.s_min
        return (x - self.d_min) / d_diff * s_diff + self.s_min

    def inverse_transform(self, x):
        d_diff = self.d_max - self.d_min
        s_diff = self.s_max - self.s_min
        return (x - self.s_min) / s_diff * d_diff + self.d_min
