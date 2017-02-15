"""Model classes"""

import numpy as np

from . import (mass, anisotropy, surface_density, volume_density,
               pdf, jeans)
from .utils import get_function, radians_per_arcsec
from .parameters import ParameterList


class DynamicalModel:
    def __init__(self, params, constants, tracers, mass_model, measurements, **kwargs):
        """Complete description of the dynamical model, including measurement model, data, and priors.

        Parameters
        ----------
        params : Parameter list object, as instantiated from config file
        constants : dictionary mapping variable names to fixed values
        tracers : list of Tracer objects
        mass_model : enclosed mass function
        measurements : list of Measurement objects
        kwargs : other keyword arguments to store for posterity
        """
        self.params = params
        self.constants = constants
        self.tracers = tracers
        self.mass_model = mass_model
        self.measurements = measurements
        self._kwargs = kwargs
        
    def __repr__(self):
        return "<{}: {}, {:d} tracers>".format(self.__class__.__name__,
                                               self.mass_model.__name__,
                                               len(self.tracers))

    def construct_kwargs(self, param_values):
        return {**self.constants, **self.params.mapping(param_values)}


    def __call__(self, param_values):
        """Log of the posterior probability"""
        
        kwargs = self.construct_kwargs(param_values)
        
        # log of the prior probability
        lnprior = self.params.lnprior(param_values)
        if not np.isfinite(lnprior):
            return -np.inf

        # log of the likelihood
        lnlike = 0
        for ll in self.measurements:
            try:
                lnlike += ll(self.mass_model, kwargs)
            except FloatingPointError as e:
                print(ll)
                print(e, "for params", param_values)
                return -np.inf
        return lnprior + lnlike

    
class Tracer:
    def __init__(self, name, anisotropy, surface_density, volume_density):
        """A dynamical tracer.  It shows some potential...

        Parameters
        ----------
        name : str, name of tracer, to be referenced by likelihood model
        anisotropy : Jeans kernel, (r, R) -> K(r, R, **kwargs)
        surface_density : R -> I(R, **kwargs)
        volume_density : r -> nu(r, **kwargs)
        """
        self.name = name
        self.anisotropy = anisotropy
        self.volume_density = volume_density
        self.surface_density = surface_density

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def __call__(self, radii, mass_model, kwargs):
        """Returns the predicted velocity dispersion at the given radii.

        Parameters
        ----------
        radii : array of projected radii, in arcsec
        mass_model : mass function, r -> M(r, **kwargs)
        kwargs : keyword arguments for all functions

        Returns
        -------
        sigma : velocity dispersion array in km/s
        """
        M = lambda r: mass_model(r, **kwargs)
        K = lambda r, R: self.anisotropy(r, R, **kwargs)
        I = lambda R: self.surface_density(R, **kwargs)
        nu = lambda r: self.volume_density(r, **kwargs)
        return jeans.sigma_jeans_interp(radii, M, K, I, nu)
        

class Measurement:

    def __init__(self, likelihood, tracers, observables):
        """Likelihood function with data.

        Parameters
        ----------
        likelihood : (sigma_jeans, *data, **kwargs) -> L(sigma_jeans, *data, *kwargs)
        tracers : list of Tracer instances
        observables : dict with keys of R, (sigma, dsigma) | (v, dv), [c, dc]
        """
        assert likelihood.__name__ in ['lnlike_continuous', 'lnlike_discrete', 'lnlike_gmm']
        self.likelihood = likelihood
        self.tracers = tracers
        self.radii = observables.pop('R')
        self.observables = observables
        

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.likelihood.__name__)

    def __call__(self, mass_model, kwargs):
        """Returns the log likelihood for these observables.

        Parameters
        ----------
        mass_model : mass function, r -> M(r, **kwargs)
        kwargs : keyword arguments for all functions

        Returns
        -------
        ll : log likelihood, in (-inf, 0)
        """
        if self.likelihood.__name__ in ['lnlike_continuous', 'lnlike_discrete']:
            sigma_jeans = self.tracers[0](self.radii, mass_model, kwargs)
            return self.likelihood(sigma_jeans, **self.observables)
        else:
            assert self.likelihood.__name__ == 'lnlike_gmm'
            # TODO, find a smarter way of dealing with GMM likelihood
            sigma_b = self.tracers[0](self.radii, mass_model, kwargs)
            sigma_r = self.tracers[1](self.radii, mass_model, kwargs)
            return self.likelihood(sigma_b, sigma_r, **self.observables, **kwargs)

        
        
        


