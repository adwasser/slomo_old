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
from . import __version__, __path__

def _version_string():
    """Get the git checksum or version number."""
    cwd = os.getcwd()
    topdir = os.path.join(__path__[0], "..")
    os.chdir(topdir)
    result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                            check=True, stdout=subprocess.PIPE)
    os.chdir(cwd)
    if result.returncode == 0:
        checksum = result.stdout.decode('utf-8').strip()
        return checksum
    return __version__


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
    return config

def create_file(hdf5_file, nwalkers, clobber=False):
    """Create a new hdf5 output file.
    hdf5_file : filename
    nwalkers : int, number of walkers for sampling
    clobber : bool, optional, if False, don't overwrite exisiting file
    """
    if clobber:
        mode = "w"
    else:
        # fail if file exists
        mode = "w-"
        
    with h5py.File(hdf5_file, mode) as f:
        
        # dump model into 
        bytestring = pickle.dumps(model)
        f["model"] = np.void(bytestring)

        # create resizable chain dataset
        ndim = len(model.params)
        f.create_dataset("chain", (nwalkers, 0, ndim),
                         maxshape=(nwalkers, None, ndim),
                         compression="lzf")

        # dump version info
        f["version"] = _version_string()
        

def read_model(hdf5_file):
    with h5py.File(hdf5_file, "r") as f:
        return pickle.loads(f['model'].values.tostring())

def read_dataset(hdf5_file, path):
    """Return a stored dataset at the specified path"""
    with h5py.File(hdf5_file, "r") as f:
        return f[path].value

def write_group(hdf5_file, group, path=""):
    """Write the group dictionary to the path on the hdf5 file"""
    with h5py.File(hdf5_file) as f:
        for key, value in group.items():
            new_path = "/".join([path, key])
            if isinstance(value, dict):
                write_group(hdf5_file, new_path, value)
            else:
                f[new_path] = value

def read_group(hdf5_file, path):
    """Return a group at the specified path as a dictionary"""
    group = {}
    with h5py.File(hdf5_file, "r") as f:
        for key, value in f[path].items():
            if isinstance(value, h5py.Dataset):
                group[key] = value.value
            elif isinstance(value, h5py.Group):
                # yay recursion!
                group[key] = read_group(hdf5_file, "/".join([path, key]))
            else:
                raise ValueError("Unknown type: " + str(value))
        return group
    
def append_to_chain(walkers, hdf5_file):
    """Walkers have shape (nwalkers, ndim)"""
    with h5py.File(hdf5_file) as f:
        chain = f["chain"]
        chain.resize((chain.shape[0], chain.shape[1] + 1, chain.shape[2]))
        chain[:, -1, :] = walkers
        f.flush()
