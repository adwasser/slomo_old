"""Probability distribution functions"""

import numpy as np
from scipy import special

def lnuniform(value, lower, upper):
    """Log of a uniform distribution.

    Parameters
    ----------
    value : float, random variable
    lower : float, lower bound
    upper : float, upper bound

    Returns
    -------
    lnp : float in (-inf, 0)
    """
    if lower < value < upper:
        return 0
    return -np.inf


def lnbeta(x, alpha, beta):
    """Log of a beta distribution.

    Parameters
    ----------
    x : float, random variable in (0, 1)
    alpha : float, higher if more weight in low values
    beta : float, higher if more weight in high values

    Returns
    -------
    lnlike : float in (-inf, 0)
    """
    term1 = (alpha - 1) * np.log(x)
    term2 = (beta - 1) * np.log(1 - x)
    term3 = -np.log(special.beta(alpha, beta))
    return term1 + term2 + term3


def lngauss(x, mu, sigma):
    """Log of gaussian likelihood distribution.

    Parameters
    ----------
    x : random variable
    mu : mean
    sigma : std

    Returns
    -------
    lnlike : float in (-inf, 0)
    """
    chi2 = ((x - mu) / sigma)**2
    norm = np.log(2 * np.pi * sigma**2)
    return -0.5 * (chi2 + norm)


def lngauss_discrete(v, dv, sigma):
    """Log of gaussian likelihood of measurements with predicted v_rms, with 
    mu = 0.
    
    Parameters
    ----------
    v : measured value
    dv : measurement uncertainty
    sigma : predicted value for std of gaussian
    
    Returns
    -------
    lnlike : float in (-inf, 0)
    """
    var = sigma**2 + dv**2
    chi2 = v**2 / var
    return -0.5 * (chi2 + np.log(2 * np.pi * var))
