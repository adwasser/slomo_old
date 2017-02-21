"""File input/output"""

import os
import subprocess
import inspect
from collections import deque

import h5py
import dill as pickle
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

def checksum():
    """Get the git checksum."""
    cwd = os.getcwd()
    gitdir, _ = os.path.split(inspect.getfile(pdf))
    os.chdir(gitdir)
    result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                            check=True, stdout=subprocess.PIPE)
    os.chdir(cwd)
    if result.returncode == 0:
        checksum = result.stdout.decode('utf-8').strip()
        return checksum


def read_yaml(filename):
    """Read in the config file and construct a model to run."""

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

def write_model(model, hdf5_file):
    bytestring = pickle.dumps(model)
    with h5py.File(hdf5_file) as f:
        f['model'] = np.void(bytestring)

        
def read_model(hdf5_file):
    with h5py.File(hdf5_file) as f:
        return pickle.loads(f['model'].values.tostring())
    
def read_chain(hdf5_file):
    """Shape is (nwalkers, niterations, ndim)"""
    with h5py.File(hdf5_file) as f:
        flatchain = f['chain'].values
    # remove walker label
    walker_labels = flatchain[:,0].flatten()
    nwalkers = int(walker_labels[-1]) + 1
    flatchain = flatchain[:,1:]
    return flatchain.reshape((nwalkers, -1, flatchain.shape[-1]))

