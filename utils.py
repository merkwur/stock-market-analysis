# https://towardsdatascience.com/introduction-to-linear-regression-in-python-c12a072bedf0
# https://github.com/raphaelvallat/antropy/blob/master/antropy/utils.py

import numpy as np
from scipy.signal import periodogram

def regression_line(X: np.ndarray, y: np.ndarray) -> np.ndarray: 
    """
    calculates the regression line from the data
    """
    # Calculate the mean of X and y
    xmean = np.mean(X)
    ymean = np.mean(y)
    # Calculate the terms needed for the numator and denominator of beta
    xy_cov = (X - xmean) * (y - ymean)
    x_var = (X -xmean)**2
    # Calculate beta and alpha
    beta = xy_cov.sum() / x_var.sum()
    alpha = ymean - (beta * xmean)
    
    return alpha + beta * X

def gain(opens: np.array, closes: np.array, window=2):
    """calculating the asset gain over the data"""
    return (closes - opens).sum()

def gradient(arr: np.ndarray) -> np.ndarray:
    """
    Calculates the gradient of the given array
    """
    return np.gradient(arr, dx=1, edge_order=2)

def _xlogx(x: np.ndarray, base=2)-> np.ndarray:
    """Returns x log_b x if x is positive, 0 if x == 0, and np.nan
    otherwise. This handles the case when the power spectrum density
    takes any zero value.
    """
    x = np.asarray(x)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    valid = x > 0
    xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(base)
    return xlogx

def spectral_entropy(arr: np.ndarray) -> float:
    """
    Calculates the spectral entropy of a signal
    """
    _, psd = periodogram(arr.to_numpy(), fs=1)
    psd_norm = psd / psd.sum(axis=0, keepdims=True)
    return -_xlogx(psd_norm).sum(axis=0)


