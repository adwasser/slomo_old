"""Routines for parsing the YAML config file"""

import numpy as np

try:
    import ruamel.yaml as yaml
except ImportError as e:
    import yaml

from .models import (Tracer, Likelihood, DynamicalModel)
from .utils import get_function
from . import mass, anisotropy, surface_density, volume_density
from .parameters import Parameter, ParameterList

tracer_models = {'anisotropy': anisotropy, 'surface_density': surface_density, 'volume_density': volume_density}

def parse(config_file):
    """Read in the config file and construct a model to run."""
    top_keys = ['params', 'constants', 'mass', 'tracers', 'likelihood', 'sampler', 'settings']
    
    with open(config_file) as f:
        config = yaml.load(f)

    # load parameter list
    param_list = config['params']
    for i, param in enumerate(param_list):
        # convert any lnprior string to dict
        if isinstance(param['lnprior'], str):
            name, args = param['lnprior'].split(',')
            args = [i for i in map(float, args)]
            
            param['lnprior'] = {'name': name, 'args': args}
        param_list[i] = Parameter(**param)
    config['params'] = ParameterList(param_list)

    # load mass
    config['mass_model'] = get_function(mass, config['mass_model'])

    # load tracers
    tracer_list = config['tracers']
    for i, tracer in enumerate(tracer_list):
        for key in tracer:
            if key == 'name':
                continue
            tracer[key] = get_function(tracer_models[key], tracer[key])
        config['tracers'][i] = Tracer(**tracer)

    # load likelihoods
    likelihood_list = config['likelihood']
    tracer_dict = {tracer.name: i for i, tracer in enumerate(config['tracers'])}
    for i, likelihood in enumerate(likelihood_list):
        # replace single tracer with list
        if not isinstance(likelihood['tracers'], list):
            likelihood['tracers'] = [likelihood['tracers']]
        likelihood['tracers'] = [config['tracers'][tracer_dict[tracer]] for tracer in likelihood['tracers']]
        # TODO: include parsing of file names, columsn for observable
        for key in likelihood['observables']:
            likelihood['observables'][key] = np.array(likelihood['observables'][key])
        config['likelihood'][i] = Likelihood(**likelihood)

    model = DynamicalModel(**config)
    return model
