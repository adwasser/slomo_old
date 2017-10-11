"""Module for Jeans kernel functions."""

import numpy as np
from scipy import special

__all__ = ["K_constant", "K_ML", "K_OM"]


def beta_inc(a, b, x):
    r"""Continuation of the incomplete beta function to allow negative values.

    :math:`B(a, b, x) = x^a a^{-1} _2F_1(a, 1 - b, a + 1, x)`

    Parameters
    ----------
    a : float
    b : float
    x : float

    Returns
    -------
    float
    """
    return x**a * a**-1 * special.hyp2f1(a, 1. - b, a + 1., x)


def K_constant(r, R, beta):
    r"""Jeans kernel for constant anisotropy parameter.

    From Mamon & Lokas 2005b, Eqn. A16

    .. note::

        `beta` must be less than 1 and cannot be exactly half integral or zero.

    Parameters
    ----------
    r : float
        The physical distance between the center and points along the
        line-of-sight.
    R : float
        The physical projected distance between the center and the 
        line-of-sight.
    beta : float
        The anisotropy parameter for a spherically symmetric system, defined as
        :math:`\beta = 1 - \sigma^2_\theta / \sigma^2_r`

    Returns
    -------
    K : float
        The value of the Jeans kernel at `r` and `R`
    """
    u = r / R
    factor = 0.5 * u**(2 * beta - 1)
    term1 = (1.5 - beta) * np.sqrt(np.pi) * \
            special.gamma(beta - 0.5) / special.gamma(beta)
    term2 = beta * beta_inc(beta + 0.5, 0.5, 1 / u**2)
    term3 = -beta_inc(beta - 0.5, 0.5, 1 / u**2)
    return factor * (term1 + term2 + term3)


def K_constant_s(r, R, beta_s, **kwargs):
    """Copy of K_constant with renamed beta parameter."""
    return K_constant(r, R, beta_s)


def K_constant_b(r, R, beta_b, **kwargs):
    """Copy of K_constant with renamed beta parameter."""
    return K_constant(r, R, beta_b)


def K_constant_r(r, R, beta_r, **kwargs):
    """Copy of K_constant with renamed beta parameter."""
    return K_constant(r, R, beta_r)


def K_ML(r, R, r_a, **kwargs):
    r"""Jeans kernel for anisotropy profile parameterization from Mamon & Lokas 2005b.

    :math:`\beta(r) =  \frac{1}{2}\frac{r}{r + r_a}`

    M&L use r_a ~ 0.18 r_virial

    Parameters
    ----------
    r : float
        The physical distance between the center and points along the
        line-of-sight.
    R : float
        The physical projected distance between the center and the 
        line-of-sight.
    r_a : float
        The scale radius of the profile.

    Returns
    -------
    K : float
        The value of the Jeans kernel at `r` and `R`
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
    r"""Jeans kernel for Osipkov-Merrit anisotropy profile.

    :math:`\beta(r) = r^2 / (r^2 + r_a^2)`

    Parameters
    ----------
    r : float
        The physical distance between the center and points along the
        line-of-sight.
    R : float
        The physical projected distance between the center and the
        line-of-sight.
    r_a : float
        The scale radius of the profile.

    Returns
    -------
    K : float
        The value of the Jeans kernel at `r` and `R`
    """
    u = r / R
    u_a = r_a / R
    factor1 = (u_a**2 + 0.5) / (u_a**2 + 1)**1.5 * (u**2 + u_a**2) / u
    factor2 = np.arctan(np.sqrt((u**2 - 1) / (u_a**2 + 1))) - 0.5 * np.sqrt(
        1 - 1 / u**2) / (u_a**2 + 1)
    return factor1 * factor2
