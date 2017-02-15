"""Run the jewels"""

import os
import subprocess
import inspect
import time

import dill as pickle
import multiprocess

import numpy as np
from emcee import EnsembleSampler

def header(model):
    """Generate header for outputs."""
    params_line = "# params: " + " ".join(model.params.names) + "\n"
    cwd = os.getcwd()
    gitdir, _ = os.path.split(inspect.getfile(model.__class__))
    result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                            check=True, stdout=subprocess.PIPE)
    if result.returncode == 0:
        checksum = result.stdout.decode('utf-8')
        return params_line + "# git checksum: " + checksum + "\n"
    return params_line


def sample(model):
    """Sample from the DynamicalModel instance."""
    
    nwalkers = model._kwargs['sampler']['nwalkers']
    ndim = len(model.params)
    threads = model._kwargs['sampler']['threads']
    if threads > 1:
        pool = multiprocess.Pool(threads)
    else:
        pool = None
    sampler = EnsembleSampler(nwalkers, ndim, model, threads=threads, pool=pool)
    settings = model.settings
    prefix = settings['prefix']
    outdir = settings['outdir']
    restart = settings['restart']
    niter = settings['niter']

    output_prefix = os.path.join(outdir, prefix)

    pickle.dump(model, open(output_prefix + '.pkl', 'wb'))
    
    if not restart:
        initial_guess = np.array(model.params._values)
        spread = 1e-4 * initial_guess
        positions = [initial_guess + spread * np.random.randn(ndim)
                     for i in range(nwalkers)]
        with open(output_prefix + '.chain', 'w') as f:
            f.write(header(model))
    else:
        # override the given inital guess positions with the last walker positions
        with open(output_prefix + '.chain', 'r') as f:
            d = deque(f, maxlen=nwalkers)
            positions = [np.array([float(s) for s in string.split()[1:]]) for string in d]
            
    count = 0
    start_time = time.time()
    for result in sampler.sample(positions, iterations=niter, storechain=False):
        print('Iteration {:4d}: {:.4e} s'.format(count + 1, time.time() - start_time))
        count += 1
        position = result[0]
        # save chain
        with open(output_prefix + '.chain', 'a') as f:
            for k in range(position.shape[0]):
                f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str, position[k]))))
    return sampler
