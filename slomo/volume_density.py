"""Density profiles
All functions with distances use "R" to refer to a projected length on the sky,
and "r" to refer to the de-projected distance from the galaxy center.  
All lengths are physical (i.e., they are actual distances, not angle subtended on the sky)
"""

import numpy as np
from scipy import special
from scipy.integrate import quad

from .utils import radians_per_arcsec
from .surface_density import b_cb

def p_ln(n):
    """'p' parameter in fitting deprojected Sersic function, 
    from Lima Neto et al. (1999)
    """
    return 1. - 0.6097 / n + 0.05463 / n**2


def nu_sersic(r, I0, Re, n, dist):
    """Sersic deprojected (3D) brightness profile approximation from
    Lima Neto et al. (1999)
    r is the deprojected radius (in arcsec) at which to evaluate the function
    I0 is the central brightness, in Lsun kpc^-2
    Re is the effective radius (at which half the luminosity is enclosed), in arcsec
    n is the Sersic index.
    dist is the distance in kpc
    """
    # distance dependent conversions
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    Re = Re * kpc_per_arcsec
    
    b = b_cb(n)
    p = p_ln(n)
    nu0 = (I0 * b**n * special.gamma(2 * n) /
           (2 * Re * special.gamma((3 - p) * n)))
    nu = nu0 * (b**n * r / Re)**(-p) * np.exp(-b_cb(n) * (r / Re)**(1 / n))
    return nu


def nu_sersic_s(r, I0_s, Re_s, n_s, dist, **kwargs):
    return nu_sersic(r, I0_s, Re_s, n_s, dist)


def nu_sersic_b(r, I0_b, Re_b, n_b, dist, **kwargs):
    return nu_sersic(r, I0_b, Re_b, n_b, dist)


def nu_sersic_r(r, I0_r, Re_r, n_r, dist, **kwargs):
    return nu_sersic(r, I0_r, Re_r, n_r, dist)


def nu_integral(r, dIdR):
    """Deprojected density profile.
    r is the deprojected radius, 
    dIdR is the derivative of surface brightness with respect to projected 
    radius, a function of R.
    """
    integrand = lambda R, r: dIdR(R) / np.sqrt(R**2 - r**2)
    try:
        size = len(r)
    except TypeError as e:
        # R is not iterable
        args = (r,)
        integral = quad(integrand, R, np.inf, args=args)[0]
    else:
        integral = np.empty(size)
        integral[:] = np.nan
        for i, radius in enumerate(r):
            args = (radius,)
            integral[i] = quad(integrand, radius, np.inf, args=args)[0]
    return - 1 / np.pi * integral


def L_sersic_tot(I0, Re, n):
    """Total luminosity of Sersic fit.
    I0 is the central brightness.
    Re is the effective radius (at which half the luminosity is enclosed).
    n is the Sersic index.
    """
    L = 2 * np.pi * n * Re**2 * I0 * special.gamma(2 * n) * b_cb(n)**(-2 * n)
    return L


def L_sersic(r, I0, Re, n):
    """Luminosity within deprojected radius r (from Lima Neto+1999).
    Note that this is NOT the same as the Graham & Driver 2005, eqn 2,
    which is for the projected radius, and thus includes more flux in the inner regions.
    """
    p = p_ln(n)
    b = b_cb(n)
    a = Re / b**n
    rho0 = I0 * special.gamma(2 * n) / (2 * a * special.gamma((3 - p) * n))
    x = (r / a)**(1 / n)
    return 4 * np.pi * rho0 * a**3 * n * special.gammainc((3 - p) * n, x) * special.gamma((3 - p) * n)


def L_R_sersic(R, I0, Re, n):
    """Sersic enclosed (<R) luminosity profile from Graham & Driver 2005
    I0 is the central brightness.
    Re is the effective radius (at which half the luminosity is enclosed).
    n is the Sersic index.
    """
    b = b_cb(n)
    x = b * (R / Re)**(1 / n)
    L = 2 * np.pi * n * I0 * Re**2 / b**(2 * n)
    return L * special.gammainc(2 * n, x) * special.gamma(2 * n)


def L_from_mag(m, M_sun, dist):
    """Gives the luminosity for a given flux at a given distance, in solar units.
    
    Parameters
    ----------
    m: apparent magnitude
    M_sun: absolute magnitude of the sun at the measured photometric band
    dist: distance to object in kpc.
    """
    return 10**((M_sun - m) / 2.5 + 4) * dist**2
