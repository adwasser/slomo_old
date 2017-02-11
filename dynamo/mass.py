"""Mass profiles"""

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


def d_n(n):
    """Einasto coefficient, as described by Retana-Montenegro+2012."""
    return (3 * n - 1 / 3. + 8 / (1215 * n) + 184 / (229635 * n**2) + 1048
            / (31000725 * n**3) - 17557576 / (1242974068875 / n**4)) # + O(n^5)


def L_sersic(r, I0, Re, n):
    """Luminosity associated with a Sersic surface density profile for a constant 
    mass-to-light ratio upsilon, at a deprojected radius, r.
    """
    p = p_ln(n)
    b = b_cb(n)
    a = Re / b**n
    rho0 = I0 * special.gamma(2 * n) / (2 * a * special.gamma((3 - p) * n))
    x = (r / a)**(1 / n)
    factor1 = 4 * np.pi * rho0 * a**3 * n
    factor2 = special.gammainc((3 - p) * n, x)
    factor3 = special.gamma((3 - p) * n)
    return = factor1 * factor2 * factor3


def M_gNFW(r, r_s, rho_s, gamma):
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
    omega = 3 - gamma
    factor1 = 4 * np.pi * rho_s * r_s**3 / omega
    factor2 = (r / r_s)**omega
    factor3 = special.hyp2f1(omega, omega, omega + 1, -r / r_s)
    return factor1 * factor2 * factor3


def M_log(r, r_c, rho_c) :
    """Cumulative Mass profile from a logarithmic (LOG) potential profile."""
    return rho_c * (3 + (r / r_c)**2) / (1 + (r / r_c)**2)**2


def M_einasto(r, h, rho0, n_einasto):
    """Mass profile for an Einasto halo."""
    M = 4 * np.pi * rho0 * h**3 * n_einasto * special.gamma(3 * n_einasto)
    return M * special.gammainc(3 * n_einasto, (r / h)**(1 / n_einasto))


def M_gNFW_L_sersic(R, r_s, rho_s, gamma, upsilon, I0_s, Re_s, n_s, dist, **kwargs):
    """gNFW halo with contant M/L and Sersic luminosity profile"""
    r = dist * R * radians_per_arcsec
    return M_gNFW(r, r_s, rho_s, gamma) + upsilon * L_sersic(r, I0_s, Re_s, n_s)


def M_gNFW_M_sersic(R, r_s, rho_s, gamma, I0_m, Re_m, n_m, dist, **kwargs):
    """gNFW halo with Sersic mass stellar mass profile"""
    r = dist * R * radians_per_arcsec
    return M_gNFW(r, r_s, rho_s, gamma) + L_sersic(r, I0_m, Re_m, n_m)
