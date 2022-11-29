# https://towardsdatascience.com/introduction-to-linear-regression-in-python-c12a072bedf0
# https://github.com/raphaelvallat/antropy/blob/master/antropy/utils.py

import numpy as np
import scipy.stats as stats
from scipy.signal import periodogram
from numpy.linalg import norm
import matplotlib.pyplot as plt

def standard_error(arr: np.ndarray) -> float:
    """s/sqrt(n)"""
    return stats.sem(arr)

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

def gain(opens: np.array, closes: np.array, window: int=2) -> float:
    """calculating the asset gain over the data"""
    return (closes - opens).sum()

def gradient(arr: np.ndarray) -> np.ndarray:
    """
    Calculates the gradient of the given array
    """
    return np.gradient(arr, dx=1, edge_order=2)

def _xlogx(x: np.ndarray, base=2) -> np.ndarray:
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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return a.dot(b) / (norm(a) * norm(b))

def sign_test(openings: np.ndarray, closings: np.ndarray) -> float:
    """Calculates the z-test. In a way, proportion of the green candles to the red candles"""
    size = len(openings)
    outrun = np.where(closings > openings, 1, 0).sum()
    return (outrun - (size / 2)) / (standard_error(openings) * .5)

def test_stats(observed: np.ndarray, expected: np.ndarray) -> float: #!!!!!!!!
    return (observed - expected) / standard_error(data)

def GM(arr):
    return stats.gmean(arr)

def entropy(openings: np.ndarray, closings: np.ndarray) -> float:
    size = len(openings)
    ratio_a = np.where(closings > openings, 1, 0).sum() / size
    ratio_b = np.where(closings < openings, 1, 0).sum() / size
    return stats.entropy([ratio_a, ratio_b], base=2)


def monte_carlo_sampling(arr: np.ndarray, n_exp: int=1000) -> float:
    mues = np.zeros(n_exp)
    for i in range(n_exp):
        sample = np.random.choice(arr, 100)
        mues[i] = sample.mean()
    
    return (np.sqrt( np.sum(np.power((mues - mues.mean()), 2))) * 100).round(6)

def PDE(mu: float, sigma: float, z: np.ndarray) -> np.ndarray:
    return (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(z-mu)**2 / (2 * sigma**2)))

def histogram(arr: np.ndarray, bins: int=42, is_plot: bool=False) -> plt or tuple[np.ndarray, np.ndarray, np.ndarray]:
    if is_plot:
        plt.style.use("Solarize_Light2")
        count, b, _ = plt.hist(arr, bins=bins, density=True)
        plt.plot(b, PDE(arr.mean(), arr.std(), b), c="orange")

        plt.show()
    else: 
        count, b = np.histogram(arr, bins=bins, density=True)
        pde = PDE(arr.mean(), arr.std(), b)

        return (count, b, pde)
    
def sample_histogram(arr: np.ndarray, bins: int=12, sample_size: int=100, is_plot: bool=False) -> plt or tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_arr = np.random.choice(arr, sample_size)
    if is_plot:
        plt.style.use("Solarize_Light2")
        count, b, _ = plt.hist(arr, bins=bins, density=True)
        plt.plot(b, PDE(arr.mean(), arr.std(), b), c="orange")

        plt.show()
    else: 
        count, b = np.histogram(arr, bins=bins, density=True)
        pde = PDE(arr.mean(), arr.std(), b)

        return (count, b, pde)    
    



