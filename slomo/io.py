"""File input/output"""

import os
import time
import subprocess
from collections import OrderedDict
import warnings

import numpy as np

import h5py
import dill as pickle
try:
    import ruamel.yaml as yaml
except ImportError as e:
    import yaml

from .models import (Tracer, Measurement, MassModel)
from .utils import get_function
from .parameters import (Parameter, ParamDict)
from . import (pdf, likelihood, transforms, mass, anisotropy, surface_density,
               volume_density)
from . import __version__, __path__

warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


def _version_string():
    """Get the git checksum or version number.

    Returns
    -------
    version : str
        If in development version, this is the git checksum.  Else, this should
        be the release version of the code.
    """
    cwd = os.getcwd()
    topdir = os.path.join(__path__[0], "..")
    os.chdir(topdir)
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], check=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        os.chdir(cwd)
        return __version__

    os.chdir(cwd)
    if result.returncode == 0:
        checksum = result.stdout.decode('utf-8').strip()
        return checksum
    return __version__


def read_yaml(filename):
    """Read in the config file and construct a dictionary from which to create
    a model.
    
    Parameters
    ----------
    filename : str
        YAML config filename

    Returns
    -------
    config : dict
         dictionary used to create a DynamicalModel
    """

    tracer_modules = {
        'anisotropy': anisotropy,
        'surface_density': surface_density,
        'volume_density': volume_density
    }

    with open(filename) as f:
        config = yaml.load(f)

    # load parameter list
    param_list = config['params']
    for i, param in enumerate(param_list):
        # convert any lnprior string to dict
        if isinstance(param['lnprior'], str):
            name, *args = map(str.strip, param['lnprior'].split(','))
            args = list(map(float, args))
            param['lnprior'] = get_function(pdf, name)
            param['lnprior_args'] = args
        # convert any transform string to a function
        try:
            transform_string = param['transform']
            param['transform'] = get_function(transforms, transform_string)
        except KeyError:
            pass
        # living dangerously... writing to list being iterated over :(
        param_list[i] = Parameter(**param)
    config['params'] = ParamDict([(p.name, p) for p in param_list])

    # load mass
    if isinstance(config['mass_model'], list):
        names, functions = zip(
            * [list(d.items())[0] for d in config['mass_model']])
    else:
        names, functions = zip(*config['mass_model'].items())
    masses = list(map(lambda f: get_function(mass, f), functions))
    mass_model = MassModel(zip(names, masses))
    config['mass_model'] = mass_model

    # load tracers
    tracer_list = config['tracers']
    for i, tracer in enumerate(tracer_list):
        for key in tracer:
            if key == 'name':
                continue
            tracer[key] = get_function(tracer_modules[key], tracer[key])
        tracer['mass_model'] = mass_model
        config['tracers'][i] = Tracer(**tracer)
    config['tracers'] = OrderedDict([(tt.name, tt)
                                     for tt in config['tracers']])

    # load likelihoods
    measurement_list = config['measurements']
    tracer_dict = {
        tracer.name: tracer
        for tracer in config['tracers'].values()
    }
    for i, measurement in enumerate(measurement_list):
        measurement['likelihood'] = get_function(likelihood,
                                                 measurement['likelihood'])
        # replace single tracer with list
        if not isinstance(measurement['model'], list):
            measurement['model'] = [measurement['model']]
        models = []
        for model in measurement['model']:
            if model in config['tracers']:
                model_function = config['tracers'][model]
            else:
                for module in tracer_modules.values():
                    try:
                        model_function = get_function(module, model)
                        break
                    except AttributeError:
                        model_function = None
                if model_function is None:
                    raise ValueError(
                        model +
                        " is not found in tracers or available functions!")
            models.append(model_function)
        measurement['model'] = models
        if isinstance(measurement['observables'], str):
            data = np.genfromtxt(
                measurement['observables'], names=True).view(np.recarray)
            measurement['observables'] = OrderedDict(
                [(name, data[name]) for name in data.dtype.names])
        config['measurements'][i] = Measurement(**measurement)
    config['measurements'] = OrderedDict([(mm.name, mm)
                                          for mm in config['measurements']])
    return config


def create_file(hdf5_file, model, nwalkers=None, clobber=False):
    """Create a new hdf5 output file.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    model : DynamicalModel
    nwalkers : int, optional
        if None then default to 10 x the number of free params
    clobber : bool, optional
        if False, don't overwrite exisiting file
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
        if nwalkers is None:
            nwalkers = 10 * ndim
        f.create_dataset(
            "chain", (nwalkers, 0, ndim),
            maxshape=(nwalkers, None, ndim),
            compression="gzip")
        # dump version info
        f["version"] = _version_string()
    write_group(hdf5_file, model._settings, "settings")


def read_model(hdf5_file):
    """Read a model file and return the DynamicalModel.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename

    Returns
    -------
    DynamicalModel
        instance of slomo.models.DynamicalModel
    """
    with h5py.File(hdf5_file, "r") as f:
        return pickle.loads(f['model'].value.tostring())


def check_model(hdf5_file):
    """Ensure that the model file will sample.

    Parameters
    ----------
    hdf5_file: str
        hdf5 filename
    """
    model = read_model(hdf5_file)
    initial_values = model.params._values
    start = time.time()
    lnp = model(initial_values)
    end = time.time()
    print("Model successfully sampled in {:.2f} s".format(end - start))


def read_dataset(hdf5_file, path):
    """Return a stored dataset at the specified path.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    path : str
        hdf5 style pathname, e.g., "settings/nwalkers"

    Returns
    -------
    dataset : any
        The dataset stored at `path`
    """
    with h5py.File(hdf5_file, "r") as f:
        return f[path].value


def write_group(hdf5_file, group, path=""):
    """Write the group dictionary to the path on the hdf5 file.

    Keys of the dictionary `group` will be new paths in the hdf5 file.
    This will recursively write to the file for nested dictionaries.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    group : dict
        dictionary to store at `path`
    path : str, optional
        hdf5 style pathname, e.g., "settings/nwalkers"
    """
    with h5py.File(hdf5_file) as f:
        for key, value in group.items():
            new_path = "/".join([path, key])
            if isinstance(value, dict):
                write_group(hdf5_file, value, new_path)
            else:
                f[new_path] = value


def read_group(hdf5_file, path):
    """Return a group at the specified path as a dictionary.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    path : str
        hdf5 style pathname, e.g., "settings"

    Returns
    -------
    group : dict
        Dictionary located at `path`
    """
    group = {}
    with h5py.File(hdf5_file, "r") as f:
        for key, value in f[path].items():
            if isinstance(value, h5py.Dataset):
                group[key] = value.value
            elif isinstance(value, h5py.Group):
                group[key] = read_group(hdf5_file, "/".join([path, key]))
            else:
                raise ValueError("Unknown type: " + str(value))
        return group


def append_to_chain(hdf5_file, walkers):
    """Add the provided walker positions to the existing chain.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    walkers : array_like
       array of walker positions of shape (nwalkers, ndim).
    """
    with h5py.File(hdf5_file) as f:
        chain = f["chain"]
        chain.resize((chain.shape[0], chain.shape[1] + 1, chain.shape[2]))
        chain[:, -1, :] = walkers
        f.flush()


def chain_shape(hdf5_file):
    """Shortcut for fetching the shape of the chain.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename

    Returns
    -------
    shape : tuple
        3-tuple of ints representing (nwalkers, niterations, ndim)
    """
    with h5py.File(hdf5_file) as f:
        chain = f["chain"]
        return chain.shape


def visit(hdf5_file):
    """Recursively visit and print all hdf5 groups.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    """
    with h5py.File(hdf5_file) as f:
        f.visititems(lambda key, value: print(key, value))
