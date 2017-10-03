"""Jeans kernels"""

import numpy as np
from scipy import special


def beta_inc(a, b, x):
    """super sketchy continuation of the incomplete beta function to allow 
    negative values
    """
    return x**a * a**-1 * special.hyp2f1(a, 1. - b, a + 1., x)


def K_constant(r, R, beta):
    """Jeans Kernel for constant anisotropy parameter.
    From Mamon & Lokas 2005b, Eqn. A16
    u = r/R, where R is the physical projected distance between the center and
    the line-of-sight, and r is the physical distance between the center and 
    points along the line-of-sight.  Beta is the anisotropy parameter, defined
    as 1 - sigma_theta**2 / sigma_r**2 for a spherically symmetric system.
    u must be greater than or equal to 1.
    Note that beta cannot be exactly half integral or zero
    """
    u = r / R
    factor = 0.5 * u**(2 * beta - 1)
    term1 = (1.5 - beta) * np.sqrt(np.pi) * \
            special.gamma(beta - 0.5) / special.gamma(beta)
    term2 = beta * beta_inc(beta + 0.5, 0.5, 1 / u**2)
    term3 = -beta_inc(beta - 0.5, 0.5, 1 / u**2)
    return factor * (term1 + term2 + term3)


def K_constant_s(r, R, beta_s, **kwargs):
    return K_constant(r, R, beta_s)


def K_constant_b(r, R, beta_b, **kwargs):
    return K_constant(r, R, beta_b)


def K_constant_r(r, R, beta_r, **kwargs):
    return K_constant(r, R, beta_r)


def K_ML(r, R, r_a, **kwargs):
    """Jeans kernel for anisotropy profile parameterization from Mamon & Lokas 2005b.
    beta(r) = 0.5 * r / (r + r_a)
    M&L use r_a ~ 0.18 r_virial
    """
    u = r / R
    u_a = r_a / R
    if u_a < 1:
        f = lambda u: np.arccos(u)
    elif u_a > 1:
        f = lambda u: np.arccosh(u)
    else:
        print('u_a == 1')
        # u_a == 1
        return (1 + 1 / u) * np.arccosh(u) - 1 / 6. * (8 / u + 7) * np.sqrt(
            (u - 1) / (u + 1))
    term1 = 0.5 / (u_a**2 - 1) * np.sqrt(1 - 1 / u**2) + (
        1 + u_a / u) * np.arccosh(u)
    term2 = -np.sign(u_a - 1) * u_a * (u_a**2 - 0.5) / (u_a**2 - 1)**1.5 * (
        1 + u_a / u) * f((u_a * u + 1) / (u + u_a))
    return term1 + term2


def K_OM(r, R, r_a, **kwargs):
    """Jeans kernel for Osipkov-Merrit anisotropy profile.
    beta(r) = r**2 / (r**2 + r_a**2)
    """
    u = r / R
    u_a = r_a / R

    factor1 = (u_a**2 + 0.5) / (u_a**2 + 1)**1.5 * (u**2 + u_a**2) / u
    factor2 = np.arctan(np.sqrt((u**2 - 1) / (u_a**2 + 1))) - 0.5 * np.sqrt(
        1 - 1 / u**2) / (u_a**2 + 1)
    return factor1 * factor2
