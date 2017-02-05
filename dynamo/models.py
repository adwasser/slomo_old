"""Model classes"""

from . import (mass, anisotropy, density, pdf, transforms, jeans)
from .utils import get_function, get_params
from .parameters import ParameterList

class Model(object):

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
    
    def __call__(self, *variables):
        """Given a dictionary of parameters names -> values
        evaluate the function.
        """
        return self._function(*variables, **self._kwargs)

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
    

class MeasurementModel(Model):
    pass


class Tracer(object):
    """It shows some potential..."""

    def __init__(self, name, observables, anisotropy, surface_density,
                 volume_density, likelihood):
        self.name = name
        self.anisotropy = AnisotropyModel(**anisotropy)
        self.surface_density = SurfaceDensityModel(**surface_density)
        self.volume_density = VolumeDensityModel(**volume_density)

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def predict_sigma(self, radii, mass_model):
        pass

    def store_kwargs(self, kwargs):
        self.anisotropy.store_kwargs(kwargs)
        self.surface_density.store_kwargs(kwargs)
        self.volume_density.store_kwarg(kwargs)
        pass

    def lnlike(self, mass_model):
        pass

    
class DynamicalModel(object):

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
        


