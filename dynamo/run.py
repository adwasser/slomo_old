"""Run the jewels"""

import os
import subprocess
import inspect
import time
from collections import deque

import dill as pickle
import multiprocess

import numpy as np
from emcee import EnsembleSampler

from . import io
from .models import DynamicalModel

def init(yaml_file, clobber=False):
    config = io.read_yaml(yaml_file)
    model = DynamicalModel(**config)
    outfile = yaml_file.split(".")[0] + ".hdf5"    
    try:
        nwalkers = model._settings['nwalkers']
    except KeyError:
        nwalkers = 10 * len(model.params)
    io.create_file(outfile, model, nwalkers=nwalkers, clobber=clobber)

def mock():
    pass

def sample(hdf5_file, niter, threads=None, mock=False):
    """Sample from the DynamicalModel instance."""
    assert io._version_string() == io.read_dataset(hdf5_file, "version")

    model = io.read_model(hdf5_file)

    settings = io.read_group(hdf5_file, "settings")
    nwalkers = settings['nwalkers']
    ndim = len(model.params)

    max_threads = multiprocess.cpu_count()
    if threads is None:
        threads = max_threads
    assert isinstance(threads, int)
    if threads < 0:
        threads = max_threads + threads
    if threads > 1:
        pool = multiprocess.Pool(threads)
    else:
        pool = None
        
    sampler = EnsembleSampler(nwalkers, ndim, model, threads=threads, pool=pool)
    
    if io.chain_shape(hdf5_file)[1] == 0:
        # starting new chain
        initial_guess = np.array(model.params._values)
        spread = 1e-4 * initial_guess
        positions = [initial_guess + spread * np.random.randn(ndim)
                     for i in range(nwalkers)]
    else:
        # override the given inital guess positions with the last walker positions
        positions = io.read_dataset(hdf5_file, "chain")[:, -1, :]
            
    count = 0
    start_time = time.time()
    for result in sampler.sample(positions, iterations=niter, storechain=False):
        print('Iteration {:4d}: {:.4e} s'.format(count + 1, time.time() - start_time))
        count += 1
        new_positions = result[0]
        io.append_to_chain(hdf5_file, new_positions)
    return sampler
