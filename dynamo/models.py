"""Model classes"""

import numpy as np

from . import (mass, anisotropy, surface_density, volume_density,
               pdf, jeans)
from .utils import radians_per_arcsec
from .parameters import Parameter, ParamDict

class DynamicalModel:
    def __init__(self, params, constants, tracers, mass_model, measurements,
                 weight_max=10, **settings):
        """Complete description of the dynamical model, including measurement model, data, and priors.

        Parameters
        ----------
        params : ParamDict object, as instantiated from config file
        constants : dictionary mapping variable names to fixed values
        tracers : OrderedDict of Tracer objects
        mass_model : enclosed mass function
        measurements : OrderedDict of Measurement objects
        settings : other keyword arguments to store for posterity
        """
        self.params = params
        self.constants = constants if constants is not None else {}
        self.tracers = tracers
        self.mass_model = mass_model
        self.measurements = measurements
        # add weight parameters
        lnprior_weight = lambda x: pdf.lnexp_truncated(x, 1, weight_max)
        for mm in self.measurements.values():
            if mm.weight:
                weight_param = Parameter("alpha_" + mm.name, 1, lnprior_weight)
                self.params[weight_param.name] = weight_param
        self._settings = settings
        
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
        for mm in self.measurements.values():
            try:
                lnlike += mm(kwargs)
            except FloatingPointError as e:
                print(mm)
                print(e, "for params", param_values)
                return -np.inf
        return lnprior + lnlike

    
class Tracer:
    def __init__(self, name, anisotropy, surface_density, volume_density, mass_model):
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
        self.mass_model = mass_model
        
    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def __call__(self, radii, kwargs):
        """Returns the predicted velocity dispersion at the given radii.

        Parameters
        ----------
        radii : array of projected radii, in arcsec
        kwargs : keyword arguments for all functions

        Returns
        -------
        sigma : velocity dispersion array in km/s
        """
        M = lambda r: self.mass_model(r, **kwargs)
        K = lambda r, R: self.anisotropy(r, R, **kwargs)
        I = lambda R: self.surface_density(R, **kwargs)
        nu = lambda r: self.volume_density(r, **kwargs)
        return jeans.sigma_jeans(radii, M, K, I, nu)
        

class Measurement:

    def __init__(self, name, likelihood, model, observables, weight=False):
        """Likelihood function with data.

        Parameters
        ----------
        name : str, name of dataset
        likelihood : (sigma_jeans, *data, **kwargs) -> L(sigma_jeans, *data, *kwargs)
        model : list of [f(R, **kwargs) -> observable]
        observables : dict with keys of R, (sigma, dsigma) | (v, dv), [c, dc] | I
        weight : if True, associate with a weight parameter for the joint likelihood
        """
        self.name = name
        self.likelihood = likelihood
        self.model = model
        self.radii = observables.pop('R')
        self.observables = observables
        self.weight = weight

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def __call__(self, kwargs):
        """Returns the log likelihood for these observables.

        Parameters
        ----------
        kwargs : keyword arguments for all functions

        Returns
        -------
        ll : log likelihood, in (-inf, 0)
        """
        try:
            weight = np.sqrt(kwargs["alpha_" + self.name])
        except KeyError:
            weight = 1
        if self.likelihood.__name__ == "lnlike_continuous":
            sigma_jeans = self.model[0](self.radii, kwargs)
            sigma = self.observables['sigma']
            dsigma = self.observables['dsigma'] / weight
            return self.likelihood(sigma_jeans, sigma, dsigma)
        elif self.likelihood.__name__ == "lnlike_discrete":
            sigma_jeans = self.model[0](self.radii, kwargs)
            v = self.observables['v']
            dv = self.observables['dv'] / weight
            return self.likelihood(sigma_jeans, v, dv)
        elif self.likelihood.__name__ == "lnlike_gmm":
            # TODO, find a smarter way of dealing with GMM likelihood
            sigma_b = self.model[0](self.radii, kwargs)
            sigma_r = self.model[1](self.radii, kwargs)
            v = self.observables['v']
            dv = self.observables['dv'] / weight
            c = self.observables['c']
            dc = self.observables['dc'] / weight
            return self.likelihood(sigma_b, sigma_r, v, dv, c, dc, **kwargs)
        elif self.likelihood.__name__ == "lnlike_density":
            I_model = self.model[0](self.radii, **kwargs)
            I = self.observables['I']
            dI = self.observables['dI'] / weight
            return self.likelihood(I_model, I, dI)
        else:
            raise ValueError(self.likelihood.__name__ + " not found!")
        
        


