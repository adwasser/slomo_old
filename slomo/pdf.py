"""Probability distribution functions"""

import numpy as np
from scipy import special


def lnuniform(value, lower, upper):
    """Log of a uniform distribution.

    Parameters
    ----------
    value : float
        random variable
    lower : float
        lower bound
    upper : float
        upper bound

    Returns
    -------
    float
        in the range (-inf, 0)
    """
    if lower < value < upper:
        return 0
    return -np.inf


def lnbeta(x, alpha, beta):
    """Log of a beta distribution.

    Parameters
    ----------
    x : float or array_like
        random variable in (0, 1)
    alpha : float
        higher if more weight in low values
    beta : float
        higher if more weight in high values

    Returns
    -------
    float or array_like
        in (-inf, 0)
    """
    term1 = (alpha - 1) * np.log(x)
    term2 = (beta - 1) * np.log(1 - x)
    term3 = -np.log(special.beta(alpha, beta))
    return term1 + term2 + term3


def lnexp(x, beta):
    """Log of an exponential distribution
    
    Parameters
    ----------
    x : float
        random variable
    beta : float
        survival parameter, inverse of rate lambda
    
    Returns
    -------
    float
    """
    if x < 0:
        return -np.inf
    return -np.log(beta) - x / beta


def lnexp_truncated(x, beta, upper):
    """Log of exponential distribution, truncated at upper limit.

    Parameters
    ----------
    x : float
        random variable
    beta : float
        survival parameter, inverse of rate lambda
    upper : float
        upper bound of distribution

    Returns
    -------
    float
    """
    if x > upper:
        return -np.inf
    return lnexp(x, beta)


def lngauss(x, mu, sigma):
    """Log of gaussian distribution.

    Parameters
    ----------
    x : float or array_like
        random variable
    mu : float
        mean
    sigma : float
        std

    Returns
    -------
    float or array_like
        in (-inf, 0)
    """
    chi2 = ((x - mu) / sigma)**2
    norm = np.log(2 * np.pi * sigma**2)
    return -0.5 * (chi2 + norm)


def lngauss_truncated(x, mu, sigma, lower, upper):
    """Log of truncated gaussian distribution.

    Parameters
    ----------
    x : float
        random variable
    mu : float
        mean
    sigma : float
        std
    lower : float
        lower bound on distribution
    upper : float
        upper bound on distribution

    Returns
    -------
    float
        in (-inf, 0)
    """
    if x < lower or x > upper:
        return -np.inf
    chi2 = ((x - mu) / sigma)**2
    norm = np.log(2 * np.pi * sigma**2)
    return -0.5 * (chi2 + norm)


def lngauss_discrete(v, dv, sigma):
    """Log of gaussian distribution for discrete tracers.

    Parameters
    ----------
    v : float or array_like
        measured value
    dv : float or array_like
        measurement uncertainty
    sigma : float or array_like
        predicted value for std of gaussian
    
    Returns
    -------
    float or array_like
    """
    std = np.sqrt(sigma**2 + dv**2)
    return lngauss(v, 0, std)
