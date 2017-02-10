"""Model classes"""

from . import (mass, anisotropy, surface_density, volume_density,
               pdf, jeans)
from .utils import get_function, get_params, radians_per_arcsec
from .parameters import ParameterList

class Model:

    def __init__(self, module, name, nvariables, replace_kwargs={}):
        """Base class for models

        module : location to look for function
        name : name of function in module
        nvariables : number of independent variables
        replace_kwargs : dictionary of keyword replacements from the default 
                         keyword names, e.g., read I0_s instead of I0
        """
        self.name = name
        self._function = get_function(module, name)
        self._nvariables = nvariables
        self._required_params = get_params(self._function)[nvariables:]
        self._replace_kwargs = replace_kwargs
        self._kwargs = {}

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)
    
    def __call__(self, *variables, **kwargs):
        """Given a dictionary of parameters names -> values
        evaluate the function.
        """
        if len(kwargs) == 0:
            return self._function(*variables, **self._kwargs)
        call_kwargs = {}
        for required_param in self._required_params:
            try:
                replaced_param = self._replace_kwargs[required_param]
            except KeyError as e:
                replaced_param = required_param
            call_kwargs[required_param] = kwargs[replaced_param]
        return self._function(*variables, **call_kwargs)

    def store_kwargs(self, kwargs):
        """Store keyword arguments from a dict of parameter names -> values"""
        for required_param in self._required_params:
            try:
                replaced_param = self._replace_kwargs[required_param]
            except KeyError as e:
                replaced_param = required_param
            self._kwargs[required_param] = kwargs[replaced_param]

            
class MassModel(Model):
    def __init__(self, name, replace_kwargs={}):
        super().__init__(mass, name, 1, replace_kwargs)
        

class AnisotropyModel(Model):
    def __init__(self, name, replace_kwargs={}):
        super().__init__(anisotropy, name, 2, replace_kwargs)

        
class SurfaceDensityModel(Model):
    def __init__(self, name, replace_kwargs={}):
        super().__init__(surface_density, name, 1, replace_kwargs)

        
class VolumeDensityModel(Model):
    def __init__(self, name, replace_kwargs={}):
        super().__init__(volume_density, name, 1, replace_kwargs)


class TracerModel(Model):
    """It shows some potential..."""
    def __init__(self, name, anisotropy, surface_density, volume_density):
        self.name = name
        self.anisotropy = AnisotropyModel(**anisotropy)
        self.surface_density = SurfaceDensityModel(**surface_density)
        self.volume_density = VolumeDensityModel(**volume_density)
        self.distance = None

    def __call__(self, radii, mass_model):
        R = radii * radians_per_arcsec * self.distance
        return sigma_jeans_interp(R, mass_model, self.anisotropy, self.surface_density, self.volume_density)
        
    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)
        
    def store_kwargs(self, kwargs):
        self.anisotropy.store_kwargs(kwargs)
        self.surface_density.store_kwargs(kwargs)
        self.volume_density.store_kwargs(kwargs)
        self.distance = kwargs['dist']

        
class LikelihoodModel(Model):
    def __init__(self, name, tracers, observables, replace_kwargs={}):
        self.name = name
        self.tracers = tracers
        self._R = observables.pop('R')
        self._observables = observables
        super().__init__(pdf, name, 1, replace_kwargs)
        
    def __call__(self, mass_model):
        ll = 0
        sigma_jeans = [tracer(self._R, mass_model) for tracer in self.tracers]
        if len(sigma_jeans) = 1:
            sigma_jeans = sigma_jeans[0]
        return super().__call__(sigma_jeans)

    def store_kwargs(self, kwargs):
        for tracer in self.tracers:
            tracer.store_kwargs(kwargs)
        super().store_kwargs({**self._observables, **kwargs})

class TracerPopulation:
    """It shows some potential..."""

    def __init__(self, name, observables, discrete, subpopulations=None,
                 anisotropy=None, surface_density=None, volume_density=None):
        self.name = name
        self.observables = observables
        self.discrete = discrete
        assert (subpopulations is not None) or (anisotropy is not None and
                                                surface_density is not None and
                                                volume_density is not None)
        self.subpopulations = {}
        if subpopulations is not None:
            if not discrete:
                raise ValueError("Can only do a mixture model on discrete tracers.")
            for pop in subpopulations:
                self.subpopulations[pop] = TracerModel(**pop)
        else:
            self.subpopulations[name] = TracerModel(name, anisotropy, surface_density, volume_density)

    def store_kwargs(self, kwargs):
        for pop in self.subpopulations:
            self.subpopulations[pop].store_kwargs
            
    def lnlike(self, mass_model):
        pass
    

class DynamicalModel:

    def __init__(self, params, constants, mass, tracers, sampler, settings):
        """Dictionaries as read off from config file."""
        self.params = ParameterList(*params)
        self.constants = constants
        self.mass = MassModel(**mass)
        self.tracers = [Tracer(t) for t in tracers]
        self.sampler = sampler
        self.settings = settings

    def store_kwargs(self, kwargs):
        self.mass.store_kwargs(kwargs)
        for t in self.tracers:
            t.store_kwargs(kwargs)
            
    def lnpost(self, proposed_values):
        """Calculate the log posterior probability for the proposed values."""
        self.params.values = proposed_values
        # log prior
        lp = self.params.lnprior
        if not np.isfinite(lp):
            return -np.inf
        kwargs = {**self.constants, **self.params.mapping}
        self.store_kwargs(kwargs)
        # log likelihood of all tracers
        ll = 0
        for t in self.tracers:
            ll += t.lnlike(self.mass)
        return ll + lp
        


