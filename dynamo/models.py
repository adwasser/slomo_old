"""Model classes"""

from . import (mass, anisotropy, density, pdf, transforms)
from .utils import get_function, get_params

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

    def store_kwargs(self, params):
        """Store keyword arguments from a dict of parameter names -> values"""
        for required_param in self._required_params:
            try:
                replaced_param = self._replace_kwargs[required_param]
            except KeyError as e:
                replaced_param = required_param
            self._kwargs[required_param] = params[replaced_param]

            
class MassModel(Model):
    def __init__(self, name, replace_kwargs={}):
        super().__init__(mass, name, 1, replace_kwargs)
        

class AnisotropyModel(Model):
    pass

class DensityModel(Model):
    pass

class Tracer(object):
    pass

class DynamicalModel(object):
    pass

