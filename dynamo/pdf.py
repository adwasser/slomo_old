"""Probability distribution functions"""

from functools import reduce
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

def lngauss_discrete(sigma, v, dv):
    var = sigma ** 2 + dv ** 2
    chi2 = v**2 / var
    ll = -0.5 * (chi2 + np.log(2 * np.pi * var))
    return ll

def lnlike_continuous(sigma_jeans, sigma, dsigma):
    return np.sum(lngauss(sigma, sigma_jeans, dsigma))


def lnlike_discrete(sigma_jeans, v, dv):
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
    var = sigma_jeans**2 + dv**2
    chi2 = v**2 / var
    ll = -0.5 * (chi2 + np.log(2 * np.pi * var))
    return np.sum(ll)


def lnlike_gmm(sigma_jeans_b, sigma_jeans_r, v, dv, c, dc,
               mu_color_b, mu_color_r, sigma_color_b, sigma_color_r, phi_b, **kwargs):
    """Gaussian mixture model likelihood

    Parameters
    ----------
    v : velocity
    dv : uncertainty in velocity
    c : color
    dc : uncertainty in color
    sigma : velocity dispersion predictions, list of length n_populations
    mu_color : mean color, list of length n_populations
    sigma_color : std of color, list of length n_populations
    phi : weights, list of length n_populations - 1
    
    Returns
    -------
    lnlike : float in (-inf, 0)
    """

    ll_b_v = lnlike_discrete(v, dv, sigma_jeans_b)
    ll_b_c = lngauss(c, mu_color_b, np.sqrt(sigma_color_b ** 2 + dc ** 2))
    ll_b = np.log(phi_b) + ll_b_v + ll_b_c

    phi_r = 1 - phi_b
    ll_r_v = lngauss_discrete(v, dv, sigma_jeans_r)
    ll_r_c = lngauss(c, mu_color_r, np.sqrt(sigma_color_r ** 2 + dc ** 2))
    ll_r = np.log(phi_r) + ll_r_v + ll_r_c

    return np.sum(np.logaddexp(ll_b, ll_r))
    

    
    
