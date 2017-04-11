"""Mass profiles"""

from collections import OrderedDict
import numpy as np
from scipy import special
from scipy import optimize

from .surface_density import b_cb
from .volume_density import p_ln
from .utils import radians_per_arcsec

def _r200(mass_function, mass_params,
          rlow=1, rhigh=10000, delta_c=200, rho_crit=137):
    """Computes r200 for the specified DM halo.
    M_dm(r) = mass_function(r, *mass_params)
    rlow and rhigh should bracket the virial radius
    """
    f = lambda r: mass_function(r, *mass_params) * 3 / (4 * np.pi * r**3) - delta_c * rho_crit
    # find the zeropoint
    rv = optimize.brentq(f, rlow, rhigh, disp=True)
    return rv


def _gNFW_to_NFW(rho_s, r_s, gamma):
    """(rho_s, r_s, gamma) -> (M200, c200)"""
    h = 0.678 # Planck 2015
    rho_crit = 277.46 * h ** 2 # Msun / kpc^3
    M = lambda r: M_gNFW(r, r_s, rho_s, gamma, dist=1 / radians_per_arcsec)
    r200 = _r200(M, (), delta_c=200, rho_crit=rho_crit)
    M200 = 4 * np.pi * r200 ** 3 / 3 * (200 * rho_crit)
    c200 = r200 / r_s
    return M200, c200
    

def d_n(n):
    """Einasto coefficient, as described by Retana-Montenegro+2012."""
    return (3 * n - 1 / 3. + 8 / (1215 * n) + 184 / (229635 * n**2) + 1048
            / (31000725 * n**3) - 17557576 / (1242974068875 / n**4)) # + O(n^5)


def heaviside_bh(R, M_bh, **kwargs):
    return M_bh


def L_sersic(r, I0, Re, n, dist):
    """Luminosity associated with a Sersic surface density profile for a constant 
    mass-to-light ratio upsilon, at a deprojected radius, r.
    Note that the scipy implementation of the incomplete gamma function includes the term 1 / Gamma(a), so that
    the standard incomplete gamma function is gamma(a, x) = special.gamma(a) * special.gammainc(a, x)
    """
    # distance dependent conversions
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    Re = Re * kpc_per_arcsec
    p = p_ln(n)
    b = b_cb(n)
    a = Re / b**n
    Ltot = 2 * np.pi * n * I0 * a ** 2 * special.gamma(2 * n)
    return Ltot * special.gammainc((3 - p) * n, (r / a) ** (1 / n))


def L_sersic_s(r, I0_s, Re_s, n_s, dist, **kwargs):
    return L_sersic(r, I0_s, Re_s, n_s, dist)


def M_gNFW(r, r_s, rho_s, gamma, dist, **kwargs):
    """Enclosed dark matter, parameterized as a generalized NFW profile.
    r is the input radius to evaluate enclosed mass.
    r_s is the scale radius.
    rho_s is the scale density.
    gamma is the shape parameter, with g = 0 for a core and g = 1 for a cusp.
    To derive the expression below from an integrated gNFW density profile, use
    the integral form of the hypergeometric function with a change of variables,
    r' -> rx, where r' is the dummy integration variable in the gNFW integrand
    and x is the dummy variable in the hypergeometric integrand.  Note that
    Beta(omega, 1) = 1 / omega.
    """
    # distance conversion
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec    
    omega = 3 - gamma
    factor1 = 4 * np.pi * rho_s * r_s**3 / omega
    factor2 = (r / r_s)**omega
    factor3 = special.hyp2f1(omega, omega, omega + 1, -r / r_s)
    return factor1 * factor2 * factor3


def M_gNFW200(r, M200, c200, gamma, dist, h=0.678, **kwargs):
    """gNFW halo parameterized with mass and concentration
    h = H0 / (100 km/s/Mpc) = 0.678 from Planck 2015
    """
    rho_crit = 277.46 * h ** 2 # Msun / kpc^3
    omega = 3 - gamma
    r200 = (3 * M200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)
    r_s = r200 / c200
    rho_s = 200 * rho_crit * omega / 3 * c200 ** gamma / special.hyp2f1(omega, omega, omega + 1, -c200)
    return M_gNFW(r, r_s, rho_s, gamma, dist, **kwargs)


def M_gNFW_dm(r, M200, gamma, dist, h=0.678, **kwargs):
    """Mass-concentration relation from Dutton & Maccio 2014"""
    c200 = 10 ** 0.905 * (M200 * h / 1e12) ** (-0.101)
    return M_gNFW200(r, M200, c200, gamma, dist, h=h, **kwargs)


def M_log(r, r_c, rho_c) :
    """Cumulative Mass profile from a logarithmic (LOG) potential profile."""
    return rho_c * (3 + (r / r_c)**2) / (1 + (r / r_c)**2)**2


def M_einasto(r, h, rho0, n_einasto):
    """Mass profile for an Einasto halo."""
    M = 4 * np.pi * rho0 * h**3 * n_einasto * special.gamma(3 * n_einasto)
    return M * special.gammainc(3 * n_einasto, (r / h)**(1 / n_einasto))


def M_sersic(r, upsilon, I0_s, Re_s, n_s, dist, **kwargs):
    return upsilon * L_sersic(r, I0_s, Re_s, n_s, dist)


def M_power(R, rho0, gamma_tot, dist, r0=1, **kwargs):
    """Power law density profile, rho = rho0 (r / r0) ^ -gamma_tot
    r0 is fixed to 1 kpc
    """
    kpc_per_arcsec = dist * radians_per_arcsec
    r = R * kpc_per_arcsec
    return 4 * np.pi * rho0 * r0 ** 3 / (3 - gamma_tot) * (r / r0) ** (3 - gamma_tot)


def M_gNFW_constant_ML(R, r_s, rho_s, gamma, upsilon, I0_s, Re_s, n_s, dist, **kwargs):
    """gNFW halo with contant M/L and Sersic luminosity profile
    R is in arcsec, converted to kpc with dist (in kpc)
    """
    return M_gNFW(R, r_s, rho_s, gamma, dist) + M_sersic(R, upsilon, I0_s, Re_s, n_s, dist)


def M_gNFW_variable_ML(R, r_s, rho_s, gamma, I0_s, Re_s, n_s, dist, **kwargs):
    """gNFW halo with Sersic mass stellar mass profile from variable IMF
    Here the sersic profile must be a mass surface density, not a luminosity
    profile.
    R is in arcsec, converted to kpc with dist (in kpc)
    """
    return M_gNFW(R, r_s, rho_s, gamma, dist) + L_sersic(R, I0_s, Re_s, n_s, dist)

def M_NFW_constant_ML(R, M200, upsilon, I0_s, Re_s, n_s, dist, **kwargs):
    return M_NFW(R, M200, dist) + upsilon * L_sersic(R, I0_s, Re_s, n_s, dist)

def M_NFW_variable_ML(R, M200, I0_s, Re_s, n_s, dist, **kwargs):
    return M_NFW(R, M200, dist) + L_sersic(R, I0_s, Re_s, n_s, dist)

def M_gNFW_dm_variable_ML(R, M200, gamma, I0_s, Re_s, n_s, dist, **kwargs):
    return M_gNFW_dm(R, M200, gamma, dist) + L_sersic(R, I0_s, Re_s, n_s, dist)

def M_gNFW200_constant_ML(R, M200, c200, gamma, upsilon, I0_s, Re_s, n_s, dist, **kwargs):
    """gNFW halo with contant M/L and Sersic luminosity profile
    R is in arcsec, converted to kpc with dist (in kpc)
    """
    return M_gNFW200(R, M200, c200, gamma, dist) + M_sersic(R, upsilon, I0_s, Re_s, n_s, dist)

def M_gNFW200_variable_ML(R, M200, c200, gamma, I0_s, Re_s, n_s, dist, **kwargs):
    """gNFW halo with Sersic mass stellar mass profile from variable IMF
    Here the sersic profile must be a mass surface density, not a luminosity
    profile.
    R is in arcsec, converted to kpc with dist (in kpc)
    """
    return M_gNFW200(R, M200, c200, gamma, dist) + L_sersic(R, I0_s, Re_s, n_s, dist)

def M_gNFW_variable_ML_bh(R, M_bh, r_s, rho_s, gamma, I0_s, Re_s, n_s, dist, **kwargs):
    return M_bh + M_gNFW_variable_ML(R, r_s, rho_s, gamma, I0_s, Re_s, n_s, dist, **kwargs)

def M_gNFW_constant_ML_bh(R, M_bh, r_s, rho_s, gamma, upsilon, I0_s, Re_s, n_s, dist, **kwargs):
    return M_bh + M_gNFW_constant_ML(R, r_s, rho_s, gamma, upsilon, I0_s, Re_s, n_s, dist, **kwargs)
