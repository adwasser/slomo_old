"""Routines for parsing the YAML config file"""

import numpy as np

try:
    import ruamel.yaml as yaml
except ImportError as e:
    import yaml

from .models import (Tracer, Measurement, DynamicalModel)
from .utils import get_function
from .parameters import (Parameter, ParameterList)
from . import (pdf, likelihood,
               mass, anisotropy,
               surface_density, volume_density)

def read(config_file):
    """Read in the config file and construct a model to run."""
    top_keys = ['params', 'constants', 'mass', 'tracers', 'measurements',
                'sampler', 'settings']
    tracer_models = {'anisotropy': anisotropy,
                     'surface_density': surface_density,
                     'volume_density': volume_density}

    with open(config_file) as f:
        config = yaml.load(f)

    # load parameter list
    param_list = config['params']
    for i, param in enumerate(param_list):
        # convert any lnprior string to dict
        if isinstance(param['lnprior'], str):
            name, *args = map(str.strip, param['lnprior'].split(','))
            args = list(map(float, args))
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
    measurement_list = config['measurements']
    tracer_dict = {tracer.name: i for i, tracer in enumerate(config['tracers'])}
    for i, measurement in enumerate(measurement_list):
        measurement['likelihood'] = get_function(likelihood, measurement['likelihood'])
        # replace single tracer with list
        if not isinstance(measurement['tracers'], list):
            measurement['tracers'] = [measurement['tracers']]
        measurement['tracers'] = [config['tracers'][tracer_dict[tracer]] for
                                  tracer in measurement['tracers']]
        if isinstance(measurement['observables'], str):
            data = np.genfromtxt(measurement['observables'], names=True).view(np.recarray)
            measurement['observables'] = {name: data[name] for name in data.dtype.names}
        config['measurements'][i] = Measurement(**measurement)

    config['sampler']['dim'] = len(config['params'])
    
    model = DynamicalModel(**config)
    return model
