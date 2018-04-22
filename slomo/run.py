"""Run the jewels"""

import time
import functools
import multiprocess
import psutil
import numpy as np
from emcee import EnsembleSampler
from tqdm import tqdm

from . import io
from .models import DynamicalModel

printf = functools.partial(print, flush=True)


def info(hdf5_file):
    """Get model information from file.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    """
    model = io.read_model(hdf5_file)
    info_str = 50 * "=" + '\n'
    info_str += "slomo\n"
    info_str += "version : {}\n".format(io.read_dataset(hdf5_file, "version"))
    info_str += "nwalkers : {}, niterations : {}, ndim : {}\n".format(
        *io.chain_shape(hdf5_file))
    info_str += 50 * "=" + '\n'
    divide = 50 * "-" + '\n'
    info_str += "\n" + divide + "Mass components\n" + divide
    for component, function in model.mass_model.items():
        info_str += "\t" + component + " : " + function.__name__ + "\n"
    info_str += "\n" + divide + "Tracers\n" + divide
    for name, tracer in model.tracers.items():
        info_str += "\t" + name + " :\n"
        info_str += "\t\tanisotropy : " + tracer.anisotropy.__name__ + "\n"
        info_str += "\t\tsurface density : " + tracer.surface_density.__name__ + "\n"
        info_str += "\t\tvolume density : " + tracer.volume_density.__name__ + "\n"
    info_str += "\n" + divide + "Measurements\n" + divide
    for name, mm in model.measurements.items():
        info_str += "\t" + name + " : " + mm.likelihood.__name__ + "\n"
    info_str += "\n" + divide + "Parameters\n" + divide
    for name, param in model.params.items():
        info_str += "\t" + name + " : " + param._lnprior.__name__
        info_str += "(x, {})\n".format(
            ", ".join(map(str, param._lnprior_args)))
    info_str += "\n" + divide + "Constants\n" + divide
    for const, value in model.constants.items():
        info_str += "\t" + const + " = " + str(value) + "\n"
    info_str += "\n" + divide + "Settings\n" + divide
    for key, value in model._settings.items():
        info_str += "\t" + key + " : " + str(value) + "\n"
    printf(info_str)


def init(yaml_file, clobber=False):
    """Create a model hdf5 file from a YAML config file.

    Parameters
    ----------
    yaml_file : str
        YAML filename
    clobber : bool, optional
        If True, then overwrite any existing hdf5 file.
    """
    config = io.read_yaml(yaml_file)
    model = DynamicalModel(**config)
    outfile = yaml_file.split(".")[0] + ".hdf5"
    try:
        nwalkers = model._settings['nwalkers']
    except KeyError:
        nwalkers = 10 * len(model.params)
    io.create_file(outfile, model, nwalkers=nwalkers, clobber=clobber)
    io.check_model(outfile)


def sample(hdf5_file, niter, threads=None):
    """Sample from the DynamicalModel instance.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    niter : int
        number of iterations
    threads : int, optional
        number of threads to use
        if None, then default to the maximum allowed threads
    """
    slomo_version = io._version_string()
    hdf5_version = io.read_dataset(hdf5_file, "version")
    if slomo_version != hdf5_version:
        printf("Version of hdf5 file ({}) doesn't match the version of slomo"
               " ({})!".format(hdf5_version, slomo_version))
    model = io.read_model(hdf5_file)

    settings = io.read_group(hdf5_file, "settings")
    nwalkers = settings['nwalkers']
    ndim = len(model.params)

    max_threads = psutil.cpu_count(logical=False)
    if threads is None:
        threads = max_threads
    threads = int(threads)
    if threads < 0:
        threads = max_threads + threads
    if threads > 1:
        pool = multiprocess.Pool(threads)
    else:
        pool = None

    sampler = EnsembleSampler(
        nwalkers, ndim, model, threads=threads, pool=pool)

    if io.chain_shape(hdf5_file)[1] == 0:
        # starting new chain
        initial_guess = np.array(model.params._values)
        # jitter for zero value guesses
        zeros = initial_guess == 0
        intial_guess[zeros] = 1e-2
        spread = 1e-4 * initial_guess
        positions = [
            initial_guess + spread * np.random.randn(ndim)
            for i in range(nwalkers)
        ]
    else:
        # override the given inital guess positions with the last walker positions
        positions = io.read_dataset(hdf5_file, "chain")[:, -1, :]

    with tqdm(total=niter) as pbar:
        for result in sampler.sample(
                positions, iterations=niter, storechain=False):
            new_positions = result[0]
            io.append_to_chain(hdf5_file, new_positions)
            pbar.update()
    return sampler
