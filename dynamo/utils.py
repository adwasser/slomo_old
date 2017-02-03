"""Miscellanea"""

import numpy as np
from astropy import constants as c
from astropy import units as u

# fail fast
np.seterr(all='raise')

radians_per_arcsec = u.arcsec.to(u.radian)
G = c.G.to(u.Msun**-1 * u.kpc * (u.km/u.s)**2).value # G in Msun-1 kpc km2 s-2
