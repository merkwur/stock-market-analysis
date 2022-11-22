import numpy as np

def regression_line(X: np.ndarray, y: np.ndarray) -> np.ndarray: 

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

