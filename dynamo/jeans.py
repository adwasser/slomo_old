"""Solving the spherical Jeans equation."""

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import interp1d

from .utils import G

def sigma_jeans(R, mass, anisotropy, I_tracer, nu_tracer):
    """Velocity dispersion in the sphereically symmetric Jeans model.
    R: an array of projected radii, in kpc
    mass: a function R -> enclosed mass, in Msun
    anisotropy: the Jeans kernel, a function r, R -> ... a number,
                where r is the deprojected radius and R is the projected radius
    I_tracer: the surface density of the tracer, R -> I, in Lsun / kpc^2
    nu_tracer: the volume density of the tracer, r -> nu, in Lsun / kpc^3
    """
    integrand = lambda r, R: anisotropy(r, R) * nu_tracer(r) * mass(r) / r
    try:
        size = len(R)
    except TypeError as e:
        # R is not iterable
        args = (R,)
        integral = quad(integrand, R, np.inf, args=args)[0]
    else:
        integral = np.empty(size)
        integral[:] = np.nan
        for i, radius in enumerate(R):
            args = (radius,)
            integral[i] = quad(integrand, radius, np.inf, args=args)[0]
    # G will convert units from Msun, kpc to km/s
    return np.sqrt(2 * G / I_tracer(R) * integral)


def sigma_jeans_interp(R, M, K, I, nu,
                       interp_points=10, cutoff_radius=1000, return_interp=False):
    """Velocity dispersion in the sphereically symmetric Jeans model.  For an array
    of input radii, the returned profile is calculated from an interpolated grid
    of a given size, distributed logrithmically across the radial range.

    Parameters
    ----------
    R: an array of projected radii, in kpc
    M : a function R -> enclosed mass, in Msun
    K : the Jeans kernel, a function r, R -> ... a number,
        where r is the deprojected radius and R is the projected radius
    I : the surface density of the tracer, R -> I, in Lsun / kpc^2
    nu : the volume density of the tracer, r -> nu, in Lsun / kpc^3
    interp_points: number of radial points (distributed logarithmically) to interpolate over
    cutoff_radius: kpc, the upper limit of the Jeans integral
    return_interp: bool, if true, return the sigma values used to interpolate
    """
    interp_radii = np.logspace(np.log10(np.amin(R)), np.log10(np.amax(R)), interp_points)
    # adjust lower and upper bounds to ensure we don't go out of interpolation domain
    interp_radii[0] -= 0.01
    interp_radii[-1] += 0.01

    integrand = lambda r, R: K(r, R) * nu(r) * M(r) / r
    size = len(interp_radii)
    integral = np.empty(size)
    integral[:] = np.nan
    for i, radius in enumerate(interp_radii):
        args = (radius,)
        integral[i] = quad(integrand, radius, cutoff_radius, args=args)[0]
        # G will convert units from Msun, kpc to km/s
        sigma = np.sqrt(2 * G / I(interp_radii) * integral)
    sigma_interp = interp1d(interp_radii, sigma)
    s = sigma_interp(R)
    if return_interp:
        return s, sigma
    return s
