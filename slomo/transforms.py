""" Functions for value transformations"""

import numpy as np

def from_log(x):
    """Common (base 10) logarithm from numpy."""
    return 10 ** x

def from_symmetrized(sym_beta):
    """The function -log(1 - beta) is symmetric in sigma_radial, sigma_tangential."""
    return 1 - 10 ** -sym_beta

def from_Mpc(dist):
    """Transform distance from Mpc to kpc"""
    return dist * 1e3
