"""Joint prior probabilities"""

import numpy as np

from colossus.cosmology import cosmology
from colossus.halo import mass_defs, mass_so
from colossus.halo.concentration import concentration
cosmo = cosmology.setCosmology('planck15')

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

def smhm(model, values, cosmo=cosmo, mdef='200c', z=0, sigma_h=0.15):
    """Joint prior from the stellar mass--halo mass relation.
    Modeled as a log-normal around the predicted stellar mass from fixed halo mass.
    Currently, the redshift dependent model of Rodriguez-Puebla is implemented.

    Parameters
    ----------
    model : DynamicalModel instance
    values : array_like
         values in parameter space at which to evaluate the prior
    cosmo : colossus.Cosmology instance
    mdef : string
        Colossus mass definition string for input halo parameters
        e.g., 'vir', '200m', '200c'
    z : float
        redshift
    """
    h = cosmo.h
    kwargs = model.construct_kwargs(values)
    if 'M200' in kwargs and 'c200' in kwargs:
        # should be '200c' def
        assert mdef == '200c'
        M200 = kwargs['M200']
        c200 = kwargs['c200']
        # colossus assumes halo masses are in Msun / h
        Mvir, rvir, cvir = mass_defs.changeMassDefinition(M200 * h, c200, z,
                                                          mdef_in=mdef,
                                                          mdef_out='vir')
        # convert back to Msun (no h scaling)
        Mvir = Mvir / h
    else:
        rho_crit = cosmo.rho_c(z) * h**2 # Msun / kpc3
        delta_c = mass_so.deltaVir(z)
        Mh_function = lambda r: model.mass_model['dm'](r, **kwargs)
        try:
            rvir = mass._rvir(Mh_function, rhigh=1e8, delta_c=delta_c, rho_crit=rho_crit)
        except ValueError:
            # meh, approximate with M200 value
            rvir = (3 * kwargs['M200'] / (4 * np.pi * 200 * rho_crit))**(1 / 3)
        Mvir = Mh_function(rvir)
    Mst = model.mass_model['st'](np.inf, **kwargs)
    if sigma_h == 'variable':
        # use the relation from Munchi+2017
        gamma = -0.26
        sigma_flat = 0.2
        logM1 = 11.5 # median value from Moster+2013
        logMvir = np.log10(Mvir)
        if logMvir < logM1:
            sigma_h = sigma_flat + gamma * (logMvir - logM1)
        else:
            sigma_h = sigma_flat
    return _ln_pdf_rp(Mvir, Mst, z=z, sigma_h=sigma_h)


def smhm_variable_scatter(model, values, cosmo=cosmo, mdef='200c', z=0):
    """Joint prior from the stellar mass--halo mass relation.
    Modeled as a log-normal around the predicted stellar mass from fixed halo mass.
    Currently, the redshift dependent model of Rodriguez-Puebla is implemented.
    
    This version uses a mass-dependent scatter from Munchi+2017.

    Parameters
    ----------
    model : DynamicalModel instance
    values : array_like
         values in parameter space at which to evaluate the prior
    cosmo : colossus.Cosmology instance
    mdef : string
        Colossus mass definition string for input halo parameters
        e.g., 'vir', '200m', '200c'
    z : float
        redshift
    """
    return smhm(model, values, cosmo=cosmo, mdef=mdef, z=z, sigma_h='variable')


def hmc(model, values, cosmo=cosmo, mdef='200c', relation='diemer15', z=0,
        sigma=0.16):
    """Joint prior from the  halo mass--concentration mass relation.

    Relation is from Diemer & Kravtsov 2015

    Parameters
    ----------
    model : DynamicalModel instance
    values : array_like
         values in parameter space at which to evaluate the prior
    cosmo : colossus.Cosmology instance
    mdef : string
        Colossus mass definition string for input halo parameters
        e.g., 'vir', '200m', '200c'
    relation : string
        See the list here for the options.
        https://bdiemer.bitbucket.io/colossus/halo_concentration.html
        Defaults to the mass-concentration model of Diemer & Kravtsov 2015.
    z : float
        redshift
    sigma : float
        scatter in M-c relation, default from Diemer & Kravtsov
    """
    h = cosmo.h
    rho_crit = cosmo.rho_c(z) * h**2 # Msun / kpc3
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
        try:
            r_s = kwargs['r_s']
            c200 = r200 / r_s
        except KeyError:
            raise ValueError('Need to have a defined halo concentration!')
    # colossus uses halo mass in units of Msun / h
    c200_model = concentration(M200 * h, mdef=mdef, z=z, model=relation)
    return pdf.lngauss(x=log_c200, mu=np.log10(c200_model), sigma=sigma)
    
