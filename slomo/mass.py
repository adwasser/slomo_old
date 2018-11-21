"""Mass profiles"""

import numpy as np
from scipy import special
from scipy import optimize

from colossus.cosmology import cosmology
from colossus.halo import mass_defs, mass_so
from colossus.halo.profile_nfw import NFWProfile
from colossus.halo.profile_einasto import EinastoProfile
from colossus.halo.concentration import concentration
cosmo = cosmology.setCosmology('planck15')

from .surface_density import b_cb
from .volume_density import p_ln
from .utils import radians_per_arcsec, G
from .colossus_profiles import SolitonNFWProfile

__all__ = [
    "L_sersic",
    "M_sersic",
    "M_NFW",
    "M_NFW200",
    "M_NFW200_dm",
    "M_gNFW",
    "M_gNFW200",
    "M_gNFW200_dm",
    "M_cNFW",
    "M_cNFW200",
    "M_cNFW_RAC",
    "M_cNFW200_RAC",
    "M_solNFW200",
    "M_log",
    "M_einasto",
    "M_burkert",
    "M_burkert_mu",
    "M_power",
    "M_point"
]

def _vir_to_fund(Mvir, cvir, z=0, mdef='200c'):
    """Computes the fundamental NFW parameters (rho_s, r_s)
    from the virial parameters (Mvir, cvir)
    
    Note that colossus uses units with:
    rhos in Msun / kpc3 * h2
    rs in kpc / h
    M in Msun / h

    Thus we need to convert back and forth from these h-scaled units
    to recover the physical units.

    Parameters
    ----------
    M200 : float or array_like
        virial mass in Msun
    c200 : float or array_like
        halo concentration
    z : float
        redshift for virial mass computation, defaults to 0
    mdef : string
        colossus virial mass definition string, defaults to '200c'
        i.e., when average density is 200 times the critical density

    Returns
    -------
    rho_s : float or array_like
        scale density in Msun / kpc3
    r_s : float or array_like
        scale radius in kpc
    """
    h = cosmo.h
    rhos, rs = NFWProfile.fundamentalParameters(M=Mvir * h, c=cvir, z=z, mdef=mdef)
    rho_s = rhos * h**2
    r_s = rs / h
    return rho_s, r_s


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


def _burkert_to_200(rho0, r0, h=0.678):
    """Convert Burkert halo parameters to M200, r200 values.

    Parameters
    ----------
    rho0 : float or array_like
        scale density in Msun / kpc^3
    r0 : float or array_like
        scale radius in kpc
    h : float, optional
        Hubble parameter in units of 100 km/s/Mpc.
        Defaults to the Planck 2015 value.

    Returns
    -------
    M200 : float or array_like
        Virial mass in Msun
    r200 : float or array_like
        halo virial radius in kpc
    """
    rho_crit = 277.46 * h**2  # Msun / kpc^3
    M = lambda r: M_burkert(r, r0, rho0, dist=1 / radians_per_arcsec)
    r200 = _rvir(M, delta_c=200, rho_crit=rho_crit)
    M200 = 4 * np.pi * r200**3 / 3 * (200 * rho_crit)
    return M200, r200


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


def M_NFW(r, r_s, rho_s, dist, **kwargs):
    """NFW profile parameterized with the scale radius and scale density.

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
    # distance conversion
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    # colossus units:
    # rhos in Msun / kpc3 * h2
    # rs in kpc / h
    # M in Msun / h
    h = cosmo.h
    return NFWProfile.M(rho_s / h**2, r_s * h, r / r_s) / h


def M_NFW200(r, M200, c200, dist, z=0, mdef='200c', **kwargs):
    """NFW profile parameterized with the virial mass and concentration.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    M200 : float
        virial mass in Msun
    c200 : float
        halo concentration
    dist : float
        distance in kpc
    z : float
        redshift for virial mass computation, defaults to 0
    mdef : string
        colossus virial mass definition string, defaults to '200c'
        i.e., when average density is 200 times the critical density

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    rho_s, r_s = _vir_to_fund(M200, c200, z=z, mdef=mdef)
    return M_NFW(r, r_s, rho_s, dist)


def M_cNFW(r, r_s, rho_s, r_c, n_c, dist, **kwargs):
    """coreNFW profile parameterized with scale radius and density,
    see Read+2016

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    r_s : float
        scale radius in kpc
    rho_s : float
        scale density in Msun / kpc^3
    r_c : float
        core radius in kpc
    n_c : float
        power-law index of core transition term
    dist : float
        distance in kpc

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    kpc_per_arcsec = dist * radians_per_arcsec
    return np.tanh(r * kpc_per_arcsec/ r_c)**n_c * M_NFW(r, r_s, rho_s, dist)


def M_cNFW200(r, M200, c200, r_c, n_c, dist, z=0, mdef='200c', **kwargs):
    """coreNFW profile parameterized with M200, c200

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    M200 : float
        virial mass in Msun
    c200 : float
        halo concentration
    r_c : float
        core radius of halo in kpc
    n_c : float
        power-law index of core transition term
    dist : float
        distance in kpc
    z : float
        redshift for virial mass computation, defaults to 0
    mdef : string
        colossus virial mass definition string, defaults to '200c'
        i.e., when average density is 200 times the critical density

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    rho_s, r_s = _vir_to_fund(M200, c200, z=z, mdef=mdef)
    return M_cNFW(r, r_s, rho_s, r_c, n_c, dist)


def M_cNFW_RAC(r, r_s, rho_s, Re_s, t_sf, dist, eta_rac=1.75, kappa_rac=0.04, **kwargs):
    """coreNFW profile, parameterized as in Read, Agartz, & Collins 2016.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    r_s : float
        scale radius in kpc
    rho_s : float
        scale density in Msun / kpc^3
    Re_s : float
        half-light radius in arcsec
    t_sf : float
        time since the start of star formation, in Gyr
    dist : float
        distance in kpc
    eta_rac : float
        eta parameter from RAC
    kappa_rac : float
        kappa parameter from RAC

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    kpc_per_arcsec = dist * radians_per_arcsec
    r_c = eta_rac * kpc_per_arcsec * Re_s
    # G in Msun^-1 kpc^3 Gyr^-2
    G_alt = 4.498502151575286e-06 
    t_dyn = 2 * np.pi * np.sqrt(r_s**3 / (G_alt * M_NFW(r_s / kpc_per_arcsec, r_s, rho_s, dist)))
    n_c = np.tanh(kappa_rac * t_sf / t_dyn)
    return M_cNFW(r, r_s, rho_s, r_c, n_c, dist)


def M_cNFW200_RAC(r, M200, c200, Re_s, t_sf, dist,
                  eta_rac=1.75, kappa_rac=0.04, z=0, mdef='200c', **kwargs):
    """coreNFW profile, parameterized as in Read, Agartz, & Collins 2016.
    NFW reparameterized to viral mass/concentration

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    M200 : float
        virial mass in Msun
    c200 : float
        halo concentration
    Re_s : float
        half-light radius in arcsec
    t_sf : float
        time since the start of star formation, in Gyr
    dist : float
        distance in kpc
    eta_rac : float
        eta parameter from RAC
    kappa_rac : float
        kappa parameter from RAC
    z : float
        redshift for virial mass computation, defaults to 0
    mdef : string
        colossus virial mass definition string, defaults to '200c'
        i.e., when average density is 200 times the critical density

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    rho_s, r_s = _vir_to_fund(M200, c200, z=z, mdef=mdef)
    return M_cNFW_RAC(r, r_s, rho_s, Re_s, t_sf, dist,
                      eta_rac=eta_rac, kappa_rac=kappa_rac)


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


def M_gNFW200(r, M200, c200, gamma, dist, z=0, mdef='200c', **kwargs):
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
    z : float
        redshift for virial mass computation, defaults to 0
    mdef : string
        colossus virial mass definition string, defaults to '200c'
        i.e., when average density is 200 times the critical density

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    rho_s, r_s = _vir_to_fund(M200, c200, z=z, mdef=mdef)
    r_vir = r_s * c200
    # re-compute rho_s to match virial mass definition
    kpc_per_arcsec = dist * radians_per_arcsec
    norm = M_gNFW(r_vir / kpc_per_arcsec, r_s=r_s, rho_s=1, gamma=gamma, dist=dist)
    rho_s = M200 / norm
    return M_gNFW(r, r_s, rho_s, gamma, dist, **kwargs)


def M_NFW200_dm(r, M200, dist, z=0, mdef='200c', **kwargs):
    """Mass-concentration relation from Dutton & Maccio 2014.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    M200 : float
        virial mass in Msun
    dist : float
        distance in kpc
    z : float
        redshift for virial mass computation, defaults to 0
    mdef : string
        colossus virial mass definition string, defaults to '200c'
        i.e., when average density is 200 times the critical density
    
    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    c200 = 10**0.905 * (M200 * h / 1e12)**(-0.101)
    return M_NFW200(r, M200, c200, dist, z=z, mdef=mdef, **kwargs)


def M_gNFW200_dm(r, M200, gamma, dist, z=0, mdef='200c', **kwargs):
    """gNFW halo parameterized with mass, with concentration from 
    Dutton & Maccio 2014.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    M200 : float
        virial mass in Msun
    gamma : float
        negative of the inner DM density log-slope
        gamma is 1 for a classic NFW cusp and 0 for a core
    dist : float
        distance in kpc
    z : float
        redshift for virial mass computation, defaults to 0
    mdef : string
        colossus virial mass definition string, defaults to '200c'
        i.e., when average density is 200 times the critical density

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    c200 = 10**0.905 * (M200 * h / 1e12)**(-0.101)
    return M_gNFW200(r, M200, c200, gamma, dist, z=z, mdef=mdef, **kwargs)


def M_solNFW200(r, M200, c200, m22, dist, rsol=None, z=0, mdef='200c',
                **kwargs):
    """Cumulative mass profile for a solition-NFW profile.

    Parameters
    ----------
    r : float or array_like
        deprojected radii in arcsec
    M200 : float
        virial mass in Msun
    c200 : float
        halo concentration
    m22 : float
        axion mass in 1e-22 eV
    dist : float
        distance in kpc
    rsol : float, optional
        soliton core radius in kpc
        if None, than use the halo mass--core radius scaling relation instead
    z : float
        redshift for virial mass computation, defaults to 0
    mdef : string
        colossus virial mass definition string, defaults to '200c'
        i.e., when average density is 200 times the critical density

    Returns
    -------
    float or array_like
        Enclosed mass in Msun
    """
    # colossus units:
    # rho in Msun / kpc3 * h2
    # r in kpc / h
    # M in Msun / h
    h = cosmo.h
    if rsol is not None:
        rsol = rsol * h
    profile = SolitonNFWProfile(M=M200 * h, c=c200, rsol=rsol, m22=m22,
                                z=z, mdef=mdef)
    # distance conversion
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    return profile.enclosedMass(r * h) / h


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


def M_plummer(r, upsilon, r_pl, ftot, dist, **kwargs):
    """Enclosed mass profile for a Plummer luminosity profile with constant
    mass-to-light ratio.


    Parameters
    ----------
    r : float or array_like
        Deprojected radius in arcsec
    upsilon : float
        mass-to-light ratio in Msun / Lsun
    r_pl : float
        Plummer radius
    ftot : float
        Total flux in Lsun / kpc2
    dist : float
        distance to the galaxy in kpc

    Returns
    -------
    float or array_like
        Enclosed Mass within deprojected radius `r`
    """
    kpc_per_arcsec = dist * radians_per_arcsec
    r = r * kpc_per_arcsec
    r_pl = r_pl * kpc_per_arcsec 
    Ltot = ftot * 4 * np.pi * dist**2
    return upsilon * Ltot * r**3 / (r**2 + r_pl**2)**1.5
    
    
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
