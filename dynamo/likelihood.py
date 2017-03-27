"""Likelihood functions"""

import numpy as np
from .pdf import (lngauss, lngauss_discrete)

def lnlike_continuous(sigma_jeans, sigma, dsigma):
    return np.sum(lngauss(sigma, sigma_jeans, dsigma))


def lnlike_density(I_model, I, dI):
    return np.sum(lngauss(I, I_model, dI))


def lnlike_discrete(sigma_jeans, v, dv):
    """Log of gaussian likelihood of measurements with predicted v_rms, with 
    mu = 0.
    """
    return np.sum(lngauss_discrete(v, dv, sigma_jeans))


def lnlike_gmm(sigma_jeans_b, sigma_jeans_r, v, dv, c, dc,
               mu_color_b, mu_color_r, sigma_color_b, sigma_color_r, phi_b, **kwargs):
    """Gaussian mixture model likelihood

    Parameters
    ----------
    sigma_jeans_b : velocity dispersion prediction for blues
    sigma_jeans_r : velocity dispersion prediction for reds
    v : velocity
    dv : uncertainty in velocity
    c : color
    dc : uncertainty in color
    mu_color_b : mean color for blues
    mu_color_r : mean color for reds
    sigma_color_b : std of color for blues
    sigma_color_r : std of color for reds
    phi_b : weight of blues (phi_r = 1 - phi_b)
    
    Returns
    -------
    lnlike : float in (-inf, 0)
    """

    ll_b_v = lngauss_discrete(v, dv, sigma_jeans_b)
    ll_b_c = lngauss(c, mu_color_b, np.sqrt(sigma_color_b ** 2 + dc ** 2))
    ll_b = np.log(phi_b) + ll_b_v + ll_b_c

    phi_r = 1 - phi_b
    ll_r_v = lngauss_discrete(v, dv, sigma_jeans_r)
    ll_r_c = lngauss(c, mu_color_r, np.sqrt(sigma_color_r ** 2 + dc ** 2))
    ll_r = np.log(phi_r) + ll_r_v + ll_r_c

    return np.sum(np.logaddexp(ll_b, ll_r))
