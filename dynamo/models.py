"""Model classes"""

import inspect

from . import (mass, anisotropy, density, pdf, transforms)

def get_function(module, name):
    """Inspect module to get the named function."""
    function = getattr(module, name)
    return function


def get_params(function):
    """Inspect function to get parameter names."""
    signature = inspect.signature(function)
    parameters = [param.name for param in signature.parameters]
    return parameters


class Parameter(object):

    def __init__(self, name, value, lnprior=None, transform=None):
        """Model parameter object

        name : str, name of parameter
        value : float, numeric value, if free param then it is the initial value
        lnprior : dict with keys (name, args), if None then fixed value parameter
        transform : name of function for transforming 
        """
        self.name = name
        self._value = value
        lnprior_function = get_function(pdf, lnprior['name'])
        if lnprior is None:
            self._lnprior = None
            self.isfree = False
        else:
            self._lnprior = lambda x: lnprior_function(x, *lnprior['args'])
            self.isfree = True
        if transform is not None:
            self.transform = get_function(transforms, transform)
        else:
            self.transform = None

    def __repr__(self):
        return "<Parameter {}>".format(self.name)
    
    @property
    def value(self):
        if self.transform is None:
            return self._value
        return self.transform(self._value)

    @property
    def lnprior(self):
        return self._lnprior(self._value)

    
class Model(object):

    def __init__(self, module, name, nvariables, args=None, kwargs=None):
        """Base class for models

        module : location to look for function
        name : name of function in module
        nvariables : number of independent variables
        args : list of argument names
        kwargs : list of keyword names
        """
        self._function = get_function(module, name)
        self._nvariables = nvariables
        self._args = args
        self._kwargs = kwargs

    def __call__(self, params, *variables):
        """Given a dictionary of parameters names -> values, evaluate the function."""
        pass

class MassModel(Model):
    def __init__(self, name, args=None, kwargs=None):
        super().__init__(mass, name="M_" + name, args, kwargs)
        

class AnisotropyModel(Model):
    pass

class DensityModel(Model):
    pass

class Tracer(object):
    pass

class DynamicalModel(object):
    pass

