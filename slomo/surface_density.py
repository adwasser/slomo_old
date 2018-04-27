"""Surface density profiles.

All functions with distances use "R" to refer to a projected length on the sky,
and "r" to refer to the de-projected distance from the galaxy center.

Functions which depend on distance will take radii as angle subtended on the
sky (e.g., in arcsec) and convert them to physical radii (e.g., kpc) at the
specified distance.
"""

import numpy as np
from scipy import special

from .utils import radians_per_arcsec

__all__ = [
    "b_cb",
    "I_sersic",
    "I_nuker",
    "I_plummer",
    "I_king"
]

def b_cb(n):
    """'b' parameter in the Sersic function, from the Ciotti & Bertin (1999)
    approximation.

    Parameters
    ----------
    n : float
        Sersic index
    
    Returns
    -------
    float
    """
    return -1. / 3 + 2. * n + 4 / (405. * n) + 46 / (25515. * n**2)


def I_nuker(R, Ib, Rb, alpha, beta, gamma):
    """Double power law density profile.
    
    Parameters
    ----------
    R : float or array_like
        projected radius in kpc
    Ib : float
        intensity at the break radius.
    Rb : float
        the break radius in kpc
    alpha : float
        the transition strength (gradual -> sharp)
    beta : float
        the outer power law index.
    gamma : float
        the inner power law index.

    Returns
    -------
    float or array_like
    """
    I = Ib * 2**((beta - gamma) / alpha) * (R / Rb)**(-gamma) * (
        1 + (R / Rb)**alpha)**((gamma - beta) / alpha)
    return I


def I_sersic(R, I0, Re, n, dist, **kwargs):
    """Sersic surface brightness profile.

    Parameters
    ----------
    R : float or array_like
        the projected radius, in arcsec
    I0 : float
        the central brightness, in Lsun kpc^-2
    Re : float
        the effective radius (at which half the luminosity is enclosed), in arcsec
    n : float
        the Sersic index.
    dist : float
        the distance in kpc
    
    Returns
    -------
    float or array_like
    """
    # distance dependent conversions
    kpc_per_arcsec = dist * radians_per_arcsec
    R = R * kpc_per_arcsec
    Re = Re * kpc_per_arcsec
    I = I0 * np.exp(-b_cb(n) * (R / Re)**(1. / n))
    return I


def I_sersic_s(R, I0_s, Re_s, n_s, dist, **kwargs):
    return I_sersic(R, I0_s, Re_s, n_s, dist)


def I_sersic_b(R, I0_b, Re_b, n_b, dist, **kwargs):
    return I_sersic(R, I0_b, Re_b, n_b, dist)


def I_sersic_r(R, I0_r, Re_r, n_r, dist, **kwargs):
    return I_sersic(R, I0_r, Re_r, n_r, dist)


def mu_sersic(R, mu_eff, Re, n):
    """Sersic surface brightness in magnitudes per square arcsec.

    Parameters
    ----------
    R : float or array_like
        the projected radius, in arcsec
    mu_eff : float
        the surface brightness at the effective radius, in mag/arcsec2
    Re : float
        the effective radius (at which half the luminosity is enclosed), in arcsec
    n : float
        the Sersic index.

    Returns
    -------
    float or array_like
    """
    return mu_eff + 2.5 * b_cb(n) / np.log(10) * ((R / Re)**(1 / n) - 1)


def I_king(R, r_c, r_lim, dist, k=1, **kwargs):
    """ King profile surface density.

    Parameters
    ----------
    R : float or array_like
        the projected radius, in arcsec
    r_c : float
        the core radius, in arcsec
    r_lim : float
        the limiting radius in arcsec
    dist : float
        the distance to the galaxy in kpc
    k : float
        the density normalization

    Returns
    -------
    float or array_like
    """
    kpc_per_arcsec = dist * radians_per_arcsec
    R = R * kpc_per_arcsec
    r_c = r_c * kpc_per_arcsec
    r_lim = r_lim * kpc_per_arcsec
    return k * ((1 + R**2 / r_c**2)**-0.5 - (1 + r_lim**2 / r_c**2)**-0.5)**2

def I_plummer(R, r_pl, dist, M=1, **kwargs):
    """Plummer profile surface density
    
    Parameters
    ----------
    R : float or array_like
        projected radius in arcsec
    r_pl : float
        Plummer radius
    dist : float
        distance to the galaxy in kpc
    M : float
        total mass in the distribution

    Returns
    -------
    float or array_like
    """
    kpc_per_arcsec = dist * radians_per_arcsec
    R = R * kpc_per_arcsec
    r_pl = r_pl * kpc_per_arcsec
    k = 3 * M / (4 * np.pi * r_pl**3)
    return M * r_pl**-2 * (1 + (R / r_pl)**2)**-2
