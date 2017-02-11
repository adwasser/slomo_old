import numpy as np

from . import (pdf, transforms)
from .utils import get_function, get_params

class Parameter:

    def __init__(self, name, value, lnprior, transform=None):
        """Model parameter object

        name : str, name of parameter
        value : float, numeric value, if free param then it is the initial value
        lnprior : dict with keys (name, args), if None then fixed value parameter
        transform : name of function for transforming 
        """
        self.name = name
        self._value = value
        lnprior_function = get_function(pdf, lnprior['name'])
        self._lnprior = lambda x: lnprior_function(x, *lnprior['args'])
        if transform is not None:
            self.transform = get_function(transforms, transform)
        else:
            self.transform = None

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)
    
    @property
    def value(self):
        if self.transform is None:
            return self._value
        return self.transform(self._value)

    @value.setter
    def value(self, x):
        self._value = x

    @property
    def lnprior(self, value):
        return self._lnprior(self._value)

    
class ParameterList:

    def __init__(self, params):
        """A list of Parameter objects"""
        if isinstance(params[0], dict):
            self._params = [Parameter(**p) for p in params]
        else:
            assert all([isinstance(p, Parameter) for p in params])
            self._params = params

    def __len__(self):
        return len(self._params)

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, value):
        self._params[key] = value
        
    def __repr__(self):
        return "<{}: {} params>".format(self.__class__.__name__, len(self))

    @property
    def names(self):
        return [p.name for p in self._params]

    @property
    def values(self):
        return [p.value for p in self._params]

    @property
    def _values(self):
        return [p._value for p in self._params]
    
    @values.setter
    def values(self, values):
        assert len(values) == len(self)
        for i, p in enumerate(self._params):
            p.value = values[i]
        
    @property
    def mapping(self):
        return {p.name: p.value for p in self._params}

    @property
    def lnprior(self):
        lp = 0
        for p in self._params:
            lp += p.lnprior
        return lp

    def _lnprior(self, values):
        assert len(values) == len(self)
        lp = 0
        for p, v in zip(self._params, values):
            lp += p._lnprior(value)
        return lp
