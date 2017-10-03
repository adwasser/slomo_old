"""Solving the spherical Jeans equation."""

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import interp1d

from .utils import G


def sigma_jeans(R,
                M,
                K,
                I,
                nu,
                interp_points=10,
                cutoff_factor=100,
                return_interp=False):
    """Velocity dispersion in the spherically symmetric Jeans model.  For an array
    of input radii, the returned profile is calculated from an interpolated grid
    of a given size, distributed logrithmically across the radial range.

    Parameters
    ----------
    R: an array of projected radii, in arcsec
    M : a function R -> enclosed mass, in Msun
    K : the Jeans kernel, a function r, R -> ... a number,
        where r is the deprojected radius and R is the projected radius
    I : the surface density of the tracer, R -> I, in count / kpc^2
    nu : the volume density of the tracer, r -> nu, in count / kpc^3
    interp_points: number of radial points (distributed logarithmically) 
                   to interpolate over; if None, than compute each point without
                   interpolation
    cutoff_factor: the upper limit of the Jeans integral, in factors of the max
                   of R
    return_interp: bool, if true, return the sigma values used to interpolate
    """
    try:
        size = len(R)
    except TypeError as e:
        R = [R]
        size = 1
        interp_points = None
    integrand = lambda r, R: K(r, R) * nu(r) * M(r) / r
    if interp_points is not None:
        radii = np.logspace(
            np.log10(np.amin(R)), np.log10(np.amax(R)), interp_points)
        # adjust lower and upper bounds to ensure we don't go out of interpolation domain
        radii[0] -= 0.01
        radii[-1] += 0.01
    else:
        radii = R
    cutoff_radius = cutoff_factor * np.amax(radii)
    size = len(radii)
    integral = np.empty(size)
    integral[:] = np.nan
    for i, radius in enumerate(radii):
        args = (radius, )
        integral[i] = quad(integrand, radius, cutoff_radius, args=args)[0]
    # G will convert units from Msun, kpc to km/s
    sigma = np.sqrt(2 * G / I(radii) * integral)
    if interp_points is None:
        return sigma
    sigma_interp = interp1d(radii, sigma)
    s = sigma_interp(R)
    if return_interp:
        return s, sigma
    return s
