"""Joint prior probabilities"""

import numpy as np

from . import pdf
from . import mass

__all__ = ["smhm", "hmc"]

def _P(x, y, z):
    return y * z - x * z / (1 + z)

def _Q(z):
    return np.exp(-4 / (1 + z)**2)

def _rp_best_fit_params(z=0):
    Q = _Q(z)
    P = lambda x, y: _P(x, y, z)
    return dict(
        epsilon = 10**(-1.758 + P(0.11, -0.061) * Q + P(-0.023, 0)),
        M0 = 10**(11.548 + P(-1.297, -0.026) * Q),
        alpha = 1.975 + P(0.714, 0.042) * Q,
        delta = 3.390 + P(-0.472, -0.931) * Q,
        gamma = 0.498 + P(-0.157, 0) * Q
    )

def _g_behroozi(x, alpha, gamma, delta):
    term1 = delta * np.log10(1 + np.exp(x))**gamma / (1 + np.exp(10**-x))
    term2 = -np.log10(10**(-alpha * x) + 1)
    return term1 + term2

def _Mstar_from_Mhalo(Mhalo, M0, alpha, gamma, delta, epsilon):
    x = np.log10(Mhalo / M0)
    term1 = np.log10(epsilon * M0)
    term2 = _g_behroozi(x, alpha, gamma, delta)
    term3 = _g_behroozi(0, alpha, gamma, delta)
    return  10**(term1 + term2 - term3)

def _ln_pdf_rp(Mhalo, Mstar, sigma_h=0.15, z=0):
    """log PDF of log-normal distribution, P(Mstar | Mhalo)"""
    params = _rp_best_fit_params(z)
    Mstar_model = _Mstar_from_Mhalo(Mhalo, **params)
    x = np.log10(Mstar)
    mu = np.log10(Mstar_model)
    return pdf.lngauss(x=x, mu=mu, sigma=sigma_h)

def smhm(model, values, h=0.678, delta_c='vir', z=0):
    """Joint prior from the stellar mass--halo mass relation.
    Modeled as a log-normal around the predicted stellar mass from fixed halo mass.
    Currently, the redshift dependent model of Rodriguez-Puebla is implemented.

    Parameters
    ----------
    model : DynamicalModel instance
    values : array_like
         values in parameter space at which to evaluate the prior
    h : float
        reduced Hubble parameter (in units of 100 km / s / Mpc)
    delta_c : float or string
        If a number than the factor of the critical density to use in the halo
        mass distribution.  Else, if delta_c == 'vir', then use the z=0 delta_vir value.
    z : float
        redshift (todo: implement redshift scaling for delta_vir...)
    """    
    rho_crit = 277.46 * h**2  # Msun / kpc^3
    if delta_c == 'vir':
        # z = 0 deltaVir from Colossus for Planck15        
        delta_c = 102.35553002960845
    kwargs = model.construct_kwargs(values)
    Mst = model.mass_model['st'](np.inf, **kwargs)
    Mh_function = lambda r: model.mass_model['dm'](r, **kwargs)
    rvir = mass._rvir(Mh_function, rhigh=1e8, delta_c=delta_c, rho_crit=rho_crit)
    Mvir = Mh_function(rvir)
    return _ln_pdf_rp(Mvir, Mst, z=z)


def hmc(model, values, h=0.678, z=0):
    """Joint prior from the  halo mass--concentration mass relation.
    Relation is for z=0 from Dutton & Maccio 2014 for an NFW profile at Delta = 200.

    Parameters
    ----------
    model : DynamicalModel instance
    values : array_like
         values in parameter space at which to evaluate the prior
    h : float
        reduced Hubble parameter (in units of 100 km / s / Mpc)
    z : float
        redshift (todo: implement redshift scaling for delta_vir...)
    """
    if z != 0:
        raise ValueError('Only valid for z = 0!')
    rho_crit = 277.46 * h**2  # Msun / kpc^3
    kwargs = model.construct_kwargs(values)
    Mh_function = lambda r: model.mass_model['dm'](r, **kwargs)
    if 'M200' in kwargs:
        M200 = kwargs['M200']
    else:
        r200 = mass._rvir(Mh_function, rhigh=1e8,
                          delta_c=200, rho_crit=rho_crit)
        M200 = Mh_function(r200)
    if 'c200' in kwargs:
        log_c200 = np.log10(kwargs['c200'])
    else:
        raise ValueError('Need to have a defined halo concentration!')
    a_dm = 0.905
    b_dm = -0.101
    sigma_dm = 0.11
    log_c200_dm = a_dm + b_dm * np.log10(M200 / (1e12 * h**-1))
    return pdf.lngauss(x=log_c200, mu=log_c200_dm, sigma=sigma_dm)
    
