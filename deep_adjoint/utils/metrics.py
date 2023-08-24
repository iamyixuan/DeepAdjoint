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
    ss_res = np.sum(np.power(true - pred, 2))
    ss_tot = np.sum(np.power(true - np.mean(true), 2))
    return 1 - ss_res / ss_tot


# def mape(true, pred):
#     return jnp.abs(true - pred).mean() / jnp.abs(true).mean() * 100


def rMSE(true, pred):
    return np.mean(np.power(true - pred, 2) / .5 * (np.power(true, 2) + np.power(pred, 2))) * 100

def sMAPE(true, pred):
    return np.mean(np.abs(true - pred) / .5 * (np.abs(true) + np.abs(pred))) * 100



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
