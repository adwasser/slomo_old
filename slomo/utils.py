"""Miscellanea

Specifies physical constants.
"""

import inspect

import numpy as np
from astropy import constants as c
from astropy import units as u

# fail fast
np.seterr(all='raise')

radians_per_arcsec = u.arcsec.to(u.radian)
G = c.G.to(u.Msun**-1 * u.kpc *
           (u.km / u.s)**2).value  # G in Msun-1 kpc km2 s-2


def get_function(module, name):
    """Inspect module to get the named function."""
    function = getattr(module, name)
    return function


def get_params(function):
    """Inspect function to get parameter names."""
    signature = inspect.signature(function)
    parameters = [param for param in signature.parameters]
    return parameters
