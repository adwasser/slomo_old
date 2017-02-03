"""Model classes"""

import inspect

from . import (mass, anisotropy, density, pdf)

def load_model(module, name):
    """Load the function from the given module."""
    function = getattr(module, name)
    signature = inspect.signature(function)
    parameters = [param.name for param in signature.parameters]
    return function, parameters


class Parameter(object):
    """Model parameter object"""
    def __init__(self, name, initial_value, lnprior_dict=None):
        """
        name : str, name of parameter
        initial_value : float, numeric value
        lnprior : dict with keys (name, args), if None then fixed parameter
        """        
        lnprior_function, lnprior_parameters = load_model(pdf, lnprior_dict['name'])
        self.lnprior = lambda x: lnprior_func(x, *lnprior['args'])
        
class Model(object):
    pass

class MassModel(Model):
    pass

class AnisotropyModel(Model):
    pass

class DensityModel(Model):
    pass

class DynamicalModel(object):
    pass

