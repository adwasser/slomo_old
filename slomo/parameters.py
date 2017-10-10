from collections import OrderedDict


def identity(x):
    return x


class Parameter:
    def __init__(self,
                 name,
                 value,
                 lnprior,
                 lnprior_args=None,
                 transform=identity):
        """Model parameter object

        name : str, name of parameter
        value : float, numeric value, if free param then it is the initial value
        lnprior : function x, *args -> log prior probability of x
        lnprior_args : list of arguments for log prior probability function
        transform : function x -> transformed parameter
        """
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
    TODO: enforce all entries being Parameter objects at init and setitem.
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