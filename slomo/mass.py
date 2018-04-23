"""Mass profiles"""

import numpy as np
from scipy import special
from scipy import optimize

from .surface_density import b_cb
from .volume_density import p_ln
from .utils import radians_per_arcsec

__all__ = [
    "L_sersic",
    "M_gNFW",
    "M_gNFW200",
    "M_NFW_dm",
    "M_log",
    "M_einasto",
    "M_burkert",
    "M_burkert_mu",
    "M_power",
    "M_sersic",
    "M_point"
]


def _rvir(mass_function,
          rlow=1,
          rhigh=10000,
          delta_c=200,
          rho_crit=137):
    """Computes the virial radius for the specified DM halo.

    Defaults to the r200 convention (i.e., the radius enclosing an
    average density equal to 200 times the critical density at z = 0).
    Uses the brentq root finding algorithm from scipy.

    Parameters
    ----------
    mass_function : function
        r -> M, in Msun
    rlow : float, optional
        lower bound for finding the virial radius, in kpc
    rhigh : float, optional
        upper bound for finding the virial radius, in kpc
    delta_c : float, optional
        factor of critical density to match to the enclosed average density
    rho_crit : float, optional
        The critical density in Msun / kpc^3

    Returns
    -------
    float
        The virial radius in kpc
    """
    f = lambda r: mass_function(r) * 3 / (4 * np.pi * r**3) - delta_c * rho_crit
    # find the zeropoint
    rv = optimize.brentq(f, rlow, rhigh, disp=True)
    return rv


def _gNFW_to_200(rho_s, r_s, gamma, h=0.678):
    """Convert gNFW parameters to M200, c200 values.

    Parameters
    ----------
    rho_s : float or array_like
        scale density in Msun / kpc^3
    r_s : float or array_like
        scale radius in kpc
    gamma : float or array_like
        negative inner density log slope
    h : float, optional
        Hubble parameter in units of 100 km/s/Mpc.
        Defaults to the Planck 2015 value.

    Returns
    -------
    M200 : float or array_like
        Virial mass in Msun
    c200 : float or array_like
        halo concentration
        :math:`c_{200} = r_{200} / r_s`
    """
    rho_crit = 277.46 * h**2  # Msun / kpc^3
    M = lambda r: M_gNFW(r, r_s, rho_s, gamma, dist=1 / radians_per_arcsec)
    r200 = _rvir(M, delta_c=200, rho_crit=rho_crit)
    M200 = 4 * np.pi * r200**3 / 3 * (200 * rho_crit)
    c200 = r200 / r_s
    return M200, c200


def d_n(n):
    """Einasto coefficient, as described by Retana-Montenegro+2012.

    Parameters
    ----------
    n : float
        Einasto index

    Returns
    -------
    float
    """
    return (3 * n - 1 / 3. + 8 / (1215 * n) + 184 / (229635 * n**2) + 1048 /
            (31000725 * n**3) - 17557576 / (1242974068875 / n**4))  # + O(n^5)


def M_point(R, M_bh, **kwargs):
    """Point mass representing a dynamically unresolved SMBH.

    Parameters
    ----------
    R : float or array_like
        projected radius in kpc
    M_bh : float
        Black hole mass in Msun
    """
    return M_bh


def L_sersic(r, I0, Re, n, dist):
    """Luminosity associated with a Sersic surface density profile for a constant
    mass-to-light ratio upsilon, at a deprojected radius, r.

    .. note::

       The scipy implementation of the incomplete gamma function includes the
       term 1 / Gamma(a), so that the standard incomplete gamma function is

       gamma(a, x) = special.gamma(a) * special.gammainc(a, x)

    Parameters
    ----------
    r : float or array_like
        Deprojected radius in arcsec
    I0 : float
        Central surface density in Lsun / kpc^2
        Note that this is a distance-independent quantity.
    Re : float
        Effecive radius, in arcsec
    n : float
        Sersic index
    dist : float
        Distance in kpc

    Returns
    -------
    float or array_like
        Enclosed luminosity within deprojected radius `r`
    """
    # distance dependent conversions
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    Re = Re * kpc_per_arcsec
    p = p_ln(n)
    b = b_cb(n)
    a = Re / b**n
    Ltot = 2 * np.pi * n * I0 * a**2 * special.gamma(2 * n)
    return Ltot * special.gammainc((3 - p) * n, (r / a)**(1 / n))


def L_sersic_s(r, I0_s, Re_s, n_s, dist, **kwargs):
    """Remapping of L_sersic keywords."""
    return L_sersic(r, I0_s, Re_s, n_s, dist)


def M_gNFW(r, r_s, rho_s, gamma, dist, **kwargs):
    """Enclosed dark matter, parameterized as a generalized NFW profile.

    .. note::
    
        To derive the expression below from an integrated gNFW density profile, use
        the integral form of the hypergeometric function with a change of variables,
        r' -> rx, where r' is the dummy integration variable in the gNFW integrand
        and x is the dummy variable in the hypergeometric integrand.  Note that
        Beta(omega, 1) = 1 / omega.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    r_s : float
        scale radius in kpc
    rho_s : float
        scale density in Msun / kpc^3
    gamma : float
        negative of the inner DM density log-slope
        gamma is 1 for a classic NFW cusp and 0 for a core
    dist : float
        distance in kpc

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    # distance conversion
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    omega = 3 - gamma
    factor1 = 4 * np.pi * rho_s * r_s**3 / omega
    factor2 = (r / r_s)**omega
    factor3 = special.hyp2f1(omega, omega, omega + 1, -r / r_s)
    return factor1 * factor2 * factor3


def M_NFW(r, r_s, rho_s, dist, **kwargs):
    """Enclosed dark matter, parameterized as a NFW profile.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    r_s : float
        scale radius in kpc
    rho_s : float
        scale density in Msun / kpc^3
    dist : float
        distance in kpc

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    return M_gNFW(r, r_s, rho_s, 1.0, dist, **kwargs)


def M_gNFW200(r, M200, c200, gamma, dist, h=0.678, **kwargs):
    """gNFW halo parameterized with mass and concentration.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    M200 : float
        virial mass in Msun
    c200 : float
        halo concentration
    gamma : float
        negative of the inner DM density log-slope
        gamma is 1 for a classic NFW cusp and 0 for a core
    dist : float
        distance in kpc
    h : float, optional
        Hubble parameter in units of 100 km/s/Mpc.
        Defaults to the Planck 2015 value.
    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    rho_crit = 277.46 * h**2  # Msun / kpc^3
    omega = 3 - gamma
    r200 = (3 * M200 / (4 * np.pi * 200 * rho_crit))**(1 / 3)
    r_s = r200 / c200
    rho_s = 200 * rho_crit * omega / 3 * c200**gamma / special.hyp2f1(
        omega, omega, omega + 1, -c200)
    return M_gNFW(r, r_s, rho_s, gamma, dist, **kwargs)


def M_NFW_dm(r, M200, dist, h=0.678, **kwargs):
    """Mass-concentration relation from Dutton & Maccio 2014.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    M200 : float
        virial mass in Msun
    dist : float
        distance in kpc
    h : float, optional
        Hubble parameter in units of 100 km/s/Mpc.
        Defaults to the Planck 2015 value.
    
    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    c200 = 10**0.905 * (M200 * h / 1e12)**(-0.101)
    return M_gNFW200(r, M200, c200, 1.0, dist, h=h, **kwargs)


def M_log(r, r_c, rho_c, dist, **kwargs):
    """Cumulative Mass profile from a logarithmic (LOG) potential profile.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    r_c : float
        core radius in kpc
    rho_c : float
        core density in Msun / kpc^3
    dist : float
        distance in kpc
    """
    # distance conversion
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    return rho_c * (3 + (r / r_c)**2) / (1 + (r / r_c)**2)**2


def M_einasto(r, h, rho0, n_einasto, dist, **kwargs):
    """Mass profile for an Einasto halo.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    h : float
        scale radius of Einasto halo in kpc
    rho0 : float
        central density in Msun / kpc^3
    n_einasto :float
        Einasto index
    dist : float
        distance in kpc

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    M = 4 * np.pi * rho0 * h**3 * n_einasto * special.gamma(3 * n_einasto)
    return M * special.gammainc(3 * n_einasto, (r / h)**(1 / n_einasto))


def M_burkert(r, r0, rho0, dist, **kwargs):
    """Mass profile for a Burkert halo.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    r0 : float
        core radius of Burkert halo in kpc
    rho0 : float
        central density in Msun / kpc^3
    dist : float
        distance in kpc
    """
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    M0 = 1.6 * rho0 * r0**3
    term1 = np.log(1 + r / r0)
    term2 = -np.arctan(r / r0)
    term3 = 0.5 * np.log(1 + (r / r0)**2)
    return 4 * M0 * (term1 + term2 + term3)


def M_burkert_mu(r, rho0, dist, mu0=10**8.15, **kwargs):
    """Mass profile for a Burkert halo with constant surface density relation.

    Default of mu0 = 10**8.15 Msun / kpc2 is taken from Donato+2009.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    rho0 : float
        central density in Msun / kpc^3
    dist : float
        distance in kpc
    mu0 : float, optional
        core surface density in Msun / kpc^2
    """
    r0 = mu0 / rho0
    return M_burkert(r, r0, rho0, dist)


def M_sersic(r, upsilon, I0_s, Re_s, n_s, dist, **kwargs):
    """Enclosed mass for a Sersic luminosity profile with constant
    mass-to-light ratio.

    Parameters
    ----------
    r : float or array_like
        Deprojected radius in arcsec
    upsilon : float
        mass-to-light ratio in Msun / Lsun
    I0_s : float
        Central surface density in Lsun / kpc^2
        Note that this is a distance-independent quantity.
    Re_s : float
        Effecive radius, in arcsec
    n_s : float
        Sersic index
    dist : float
        Distance in kpc

    Returns
    -------
    float or array_like
        Enclosed Mass within deprojected radius `r`
    """
    return upsilon * L_sersic(r, I0_s, Re_s, n_s, dist)


def M_power(r, rho0, gamma_tot, dist, r0=1, **kwargs):
    r"""Power law density profile.

    :math:`\rho = \rho_0 (r / r_0) ^{-\gamma_\mathrm{tot}}`

    .. note::

        :math:`r_0` must be fixed since it is degenerate with :math:`\rho_0`.

    Parameters
    ----------
    r : float or array_like
        Deprojected radius in arcsec
    rho0 : float
        density at r0, in Msun / kpc^3
    gamma_tot : float
        negative of the total mass density log-slope
    dist : float
        Distance in kpc
    r0 : float, optional
        scale radius in kpc, defaults to 1 kpc

    Returns
    -------
    float or array_like
        Enclosed Mass within deprojected radius `r`

    """
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    return 4 * np.pi * rho0 * r0**3 / (3 - gamma_tot) * (r / r0)**(
        3 - gamma_tot)
