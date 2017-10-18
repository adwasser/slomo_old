from collections import OrderedDict


def _identity(x):
    """A silly way of setting a default transform."""
    return x


class Parameter:
    """Model parameter.

    This class keeps track of priors for parameters and parameter transforms.
    The idea is that the sampler wants to keeps track of parameters in some
    coordinate space (e.g., the log of a scale parameter), while the dynamical
    model wants to see physical values.  The transform function will map values
    from the sampled space to the physical space.

    Parameters
    ----------
    name : str
        name of parameter
    value : float
        numeric value, the initial value
    lnprior : function
        x, \*args -> log prior probability of x
    lnprior_args : list, optional
        arguments for log prior probability function
    transform : function, optional
        sampled parameter -> physical parameter

    Attributes
    ----------
    value : float
        if fetched, it will be in physical space
        if set, it will be as the sampled space

    Methods
    -------
    lnprior(value)
        Return the value of the log prior at the given point
    """
    def __init__(self,
                 name,
                 value,
                 lnprior,
                 lnprior_args=None,
                 transform=_identity):
        self.name = name
        # parameter value as seen by sampler
        self._value = value
        self._lnprior = lnprior
        if lnprior_args is None:
            self._lnprior_args = ()
        else:
            self._lnprior_args = lnprior_args
        self.transform = transform

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    @property
    def value(self):
        # parameter value as seen by dynamical model
        return self.transform(self._value)

    @value.setter
    def value(self, x):
        self._value = x

    def lnprior(self, value):
        return self._lnprior(value, *self._lnprior_args)


class ParamDict(OrderedDict):
    """OrderedDict of Parameter objects

    Attributes
    ----------
    names : list
        list of parameter names (as str)

    Methods
    -------
    lnprior(values)
        Map a list of values to the sum of the log priors of each parameter.
    mapping(values)
        Create a dictionary mapping the parameter name to the physical value.
    index(name)
        Return an integer giving the index of the parameter with the specified
        name.
    """

    @property
    def names(self):
        return list(self.keys())

    @property
    def _values(self):
        return [p._value for p in self.values()]

    def lnprior(self, values):
        # pre-transformed values
        assert len(values) == len(self)
        lp = 0
        for value, param in zip(values, self.values()):
            lp += param.lnprior(value)
        return lp

    def mapping(self, values):
        """Construct a dictionary from parameter name to assigned values.
        If the parameter has a transformation from the sampled parameter to the
        physical parameter, that is performed for the output."""
        assert len(values) == len(self)
        return {p.name: p.transform(v) for v, p in zip(values, self.values())}

    def index(self, name):
        """Get the index (unique if exists) for the given name."""
        return list(self.keys()).index(name)
