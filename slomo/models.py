"""Model classes"""

from collections import OrderedDict

import numpy as np

from . import (pdf, jeans)
from .parameters import Parameter


class DynamicalModel:
    """Complete description of the dynamical model, including measurement
    model, data, and priors.
    
    Parameters
    ----------
    params : ParamDict
    constants : dict
        Mapping of variable names to fixed values
    tracers : OrderedDict
        Contains Tracer instances
    mass_model : MassModel
        Functions for enclosed mass
    measurements : OrderedDict
        Contains Measurement instances
    joint_priors : list
        list of strings denoting which, if any, joint priors to use
        currently a stellar-to-halo mass relation ("smhm") and a 
        halo mass-concentration relation ("hmc") are offered
    settings : dict
        Other keyword arguments to store for posterity
    """
    def __init__(self,
                 params,
                 constants,
                 tracers,
                 mass_model,
                 measurements,
                 weight_max=10,
                 joint_priors=None,
                 **settings):
        self.params = params
        self.constants = constants if constants is not None else {}
        self.tracers = tracers
        self.mass_model = mass_model
        self.measurements = measurements
        self.joint_priors = joint_priors
        # add weight parameters
        for mm in self.measurements.values():
            if mm.weight:
                weight_param = Parameter(
                    "alpha_" + mm.name,
                    value=1,
                    lnprior=pdf.lnexp_truncated,
                    lnprior_args=(1, weight_max))
                self.params[weight_param.name] = weight_param
        self._settings = settings

    def __repr__(self):
        fmt_str = "<{}: {:d} mass components, {:d} tracers>"
        return fmt_str.format(self.__class__.__name__,
                              len(self.mass_model), len(self.tracers))

    def construct_kwargs(self, param_values):
        """Takes a list of parameter values and returns a mapping of names to
        values.

        Parameters
        ----------
        param_values : iterable
        
        Returns
        -------
        dict
            names (str) -> values (float)
        """
        return {**self.constants, **self.params.mapping(param_values)}

    def __call__(self, param_values):
        """Log of the posterior probability.
        
        Parameters
        ----------
        param_values : list
            list of parameter values (floats) in the order taken from the params
            OrderedDict

        Returns
        -------
        float
            Log of the posterior probability at the specified point in parameter
            space.
        """

        kwargs = self.construct_kwargs(param_values)
        # build mass interpolation table
        self.mass_model.build_interp_table(**kwargs)
        
        # log of the prior probability
        lnprior = self.params.lnprior(param_values)
        if not np.isfinite(lnprior):
            return -np.inf
        if self.joint_priors is not None:
            for f in self.joint_priors:
                lnprior += f(self, param_values)
        
        # log of the likelihood
        lnlike = 0
        for mm in self.measurements.values():
            try:
                lnlike += mm(kwargs)
            except FloatingPointError as e:
                print(mm)
                print(e, "for params", param_values)
                return -np.inf
        return lnprior + lnlike


class MassModel(OrderedDict):
    """Multi-component mass model.

    Parameters
    ----------
    components : iterable
        Sequence of (key, function) tuples where function refers to a mapping of
        r (in arcsec), \*\*kwargs -> M (in Msun)
    """
    def __init__(self, *args, rgrid=None, **kwargs):
        self.rgrid = rgrid
        self.Mgrid = None
        self.interp_kwargs = None
        super().__init__(*args, **kwargs)
        

    def build_interp_table(self, rgrid=None, **kwargs):
        """Cache the total mass profile across the radius range.

        Parameters
        ----------
        rmin : float
            minimum radius (arcsec)
        rmax : float
            maximum radius (arcsec)
        n : int, optional
            number of steps in interpolation table
        kwargs : dict
            keyword arguments for mass models
        """
        if rgrid is not None:
            self.rgrid = rgrid
        elif self.rgrid is None:
            raise ValueError("Must define the interpolation grid by passing rgrid.")
        self.Mgrid = sum([M(self.rgrid, **kwargs) for M in self.values()])
        self.interp_kwargs = kwargs

        
    def _kwargs_match(self, **kwargs):
        return all([self.interp_kwargs[key] == value
                    for key, value in kwargs.items()])

    
    def __call__(self, radii, interpolate=True, rgrid=None, **kwargs):
        """Calculate the enclosed mass of all components.

        Parameters
        ----------
        radii : float or array_like
            radii (in arcsec) at which to compute the mass
        interpolate : bool, optional
            if true, use an interpolation table
        rgrid : array_like, optional
            override any built-in interpolation grid
        kwargs : dict
            mapping of names (str) -> values (float)

        Returns
        -------
        float or array_like
            Enclosed mass in Msun
        """
        if interpolate:
            if (not self._kwargs_match(**kwargs)) or (self.Mgrid is None):
                self.build_interp_table(rgrid=rgrid, **kwargs)
            return np.interp(radii, self.rgrid, self.Mgrid)
        return sum([M(radii, **kwargs) for M in self.values()])


class Tracer:
    """A dynamical tracer.  It shows some potential...
    
    Parameters
    ----------
    name : str
        name of tracer, to be referenced by likelihood model
    anisotropy : function
        Jeans kernel, (r, R) -> K(r, R, \*\*kwargs)
    surface_density : function
        R -> I(R, \*\*kwargs)
    volume_density : function
        r -> nu(r, \*\*kwargs)
    """
    def __init__(self, name, anisotropy, surface_density, volume_density,
                 mass_model, aperture=False):
        self.name = name
        self.anisotropy = anisotropy
        self.volume_density = volume_density
        self.surface_density = surface_density
        self.mass_model = mass_model
        self.aperture = aperture

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def __call__(self, radii, kwargs, interp_points=10):
        """Returns the predicted velocity dispersion at the given radii.

        Parameters
        ----------
        radii : float or array_like
            projected radii, in arcsec
        kwargs : dict
            keyword arguments for all functions

        Returns
        -------
        sigma : float
            velocity dispersion array in km/s
        """
        M = lambda r: self.mass_model(r, **kwargs)
        K = lambda r, R: self.anisotropy(r, R, **kwargs)
        I = lambda R: self.surface_density(R, **kwargs)
        nu = lambda r: self.volume_density(r, **kwargs)
        if self.aperture:
            return jeans.sigma_aperture(radii, M, K, I, nu, interp_points=interp_points)
        return jeans.sigma_jeans(radii, M, K, I, nu, interp_points=interp_points)


class Measurement:
    """Likelihood function with data.

    Parameters
    ----------
    name : str
        name of dataset
    likelihood : function
        (sigma_jeans, \*data, \*\*kwargs) -> L(sigma_jeans, \*data, \*kwargs)
    model : list
        sequence of model functions, f(R, \*\*kwargs) -> observable
    observables : dict
        has keys of R, (sigma, dsigma) | (v, dv), [c, dc] | I
    weight : bool
        if True, associate with a weight parameter for the joint likelihood
    """
    def __init__(self, name, likelihood, model, observables, weight=False):
        self.name = name
        self.likelihood = likelihood
        self.model = model
        self.radii = observables.pop('R')
        self.observables = observables
        self.weight = weight

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def __call__(self, kwargs):
        """Returns the log likelihood for these observables.

        Parameters
        ----------
        kwargs : dict
            keyword arguments for all functions

        Returns
        -------
        ll : float
            log likelihood, in (-inf, 0)
        """
        try:
            weight = np.sqrt(kwargs["alpha_" + self.name])
        except KeyError:
            weight = 1
        if self.likelihood.__name__ == "lnlike_continuous":
            sigma_jeans = self.model[0](self.radii, kwargs)
            sigma = self.observables['sigma']
            dsigma = self.observables['dsigma'] / weight
            return self.likelihood(sigma_jeans, sigma, dsigma)
        elif self.likelihood.__name__ == "lnlike_discrete":
            sigma_jeans = self.model[0](self.radii, kwargs)
            v = self.observables['v']
            dv = self.observables['dv'] / weight
            return self.likelihood(sigma_jeans, v, dv, **kwargs)
        elif self.likelihood.__name__ == "lnlike_discrete_outlier":
            sigma_jeans = self.model[0](self.radii, kwargs)
            v = self.observables['v']
            dv = self.observables['dv'] / weight
            return self.likelihood(sigma_jeans, v, dv, **kwargs)
        elif self.likelihood.__name__ == "lnlike_gmm":
            # TODO, find a smarter way of dealing with GMM likelihood
            sigma_b = self.model[0](self.radii, kwargs)
            sigma_r = self.model[1](self.radii, kwargs)
            v = self.observables['v']
            dv = self.observables['dv'] / weight
            c = self.observables['c']
            dc = self.observables['dc'] / weight
            return self.likelihood(sigma_b, sigma_r, v, dv, c, dc, **kwargs)
        elif self.likelihood.__name__ == "lnlike_density":
            I_model = self.model[0](self.radii, **kwargs)
            I = self.observables['I']
            dI = self.observables['dI'] / weight
            return self.likelihood(I_model, I, dI)
        else:
            raise ValueError(self.likelihood.__name__ + " not found!")
