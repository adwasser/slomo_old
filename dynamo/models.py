"""Model classes"""

from . import (mass, anisotropy, surface_density, volume_density,
               pdf, jeans)
from .utils import get_function, get_params, radians_per_arcsec
from .parameters import ParameterList


class DynamicalModel:
    def __init__(self, params, constants, tracers, mass, likelihood):
        """Complete description of the dynamical model, including measurement model, data, and priors.

        Parameters
        ----------
        params : Parameter list object, as instantiated from config file
        constants : dictionary mapping variable names to fixed values
        tracers : list of Tracer objects
        mass : enclosed mass function
        likelihood : dictionary describing measurement model, as instantiated from config file
        """
        self.params = params
        self.constants = constants
        self.tracers = tracers
        self.mass = mass
        self.likelihood = likelihood

        
    def __repr__(self):
        return "<{}: {}, {:d} tracers>".format(self.__class__.__name__, self.mass.__name__, len(self.tracers))


    def lnpost(self, param_values):
        """Log of the posterior probability"""

        #TODO, define kwargs from constants and params
        new_params = self.params.mapping.update(asdfasdfasdfjasdf) ########################## FIX THIS
        kwargs = {**self.constants, **new_params}
        
        # log of the prior probability
        lnprior = self.params._lnprior(param_values)
        if not np.isfinite(lnprior):
            return -np.inf

        # log of the likelihood
        lnlike = self.likelihood(mass, kwargs)
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
        self.anisotropy = anisotropy
        self.volume_density = volume_density
        self.surface_density = surface_density

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def __call__(self, radii, mass, kwargs):
        """Returns the predicted velocity dispersion at the given radii.

        Parameters
        ----------
        radii : array of projected radii, in arcsec
        mass : mass function, r -> M(r, **kwargs)
        kwargs : keyword arguments for all functions

        Returns
        -------
        sigma : velocity dispersion array in km/s
        """
        M = lambda r: mass(r, **kwargs)
        K = lambda r, R: self.anisotropy(r, R, **kwargs)
        I = lambda R: self.surface_density(R, **kwargs)
        nu = lambda r: self.volume_density(r, **kwargs)
        return jeans.sigma_jeans_interp(radii, M, K, I, nu)
        

class Likelihood:
    def __init__(self, likelihood, tracers, observables):
        """Likelihood function with data.

        Parameters
        ----------
        likelihood : (sigma_jeans, *data, **kwargs) -> L(sigma_jeans, *data, *kwargs)
        tracers : list of Tracer instances
        observables : dict with keys of R, (sigma, dsigma) | (v, dv), [c, dc]
        """
        self.likelihood = likelihood
        self.tracers = tracers
        self.radii = observables.pop('R')
        self.observables = observables
        

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.likelihood.__name__)

    def __call__(self, mass, kwargs):
        if likelihood.__name__ in ['lnlike_continuous', 'lnlike_discrete']:
            sigma_jeans = tracers[0](R, mass, kwargs)
            return self.likelihood(sigma_jeans, **self.observables)
        elif likelihood.__name__ == 'lnlike_gmm':
            # TODO, find a smarter way of dealing with GMM likelihood
            sigma_b = tracers[0](R, mass, kwargs)
            sigma_r = tracers[1](R, mass, kwargs)
            return self.likelihood(sigma_b, sigma_r, **self.observables, **kwargs)

        
        
        


