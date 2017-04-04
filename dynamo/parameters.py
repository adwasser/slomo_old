import numpy as np

from . import (pdf, transforms)
from .utils import get_function, get_params

class Parameter:

    def __init__(self, name, value, lnprior, lnprior_args=[], transform=None):
        """Model parameter object

        name : str, name of parameter
        value : float, numeric value, if free param then it is the initial value
        lnprior : function x -> log prior probability of x
        transform : name of function for transforming 
        """
        self.name = name
        self._value = value
        self._lnprior = lnprior
        self._lnprior_args = lnprior_args
        if transform is not None:
            self.transform = get_function(transforms, transform)
        else:
            self.transform = lambda x: x

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)
    
    @property
    def value(self):
        return self.transform(self._value)

    @value.setter
    def value(self, x):
        self._value = x

    @property
    def lnprior(self, value):
        return self._lnprior(self._value, *self._lnprior_args)

    
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

    def append(self, param):
        self._params.append(param)
        
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
        
    def lnprior(self, values):
        # pre-transformed values
        assert len(values) == len(self)
        lp = 0
        for p, v in zip(self._params, values):
            lp += p._lnprior(v, *p._lnprior_args)
        return lp

    def mapping(self, values):
        # pre-transformed values
        assert len(values) == len(self)
        return {p.name: p.transform(v) for p, v in zip(self._params, values)}
