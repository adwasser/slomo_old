# slomo

Spherical Jeans modeling of slow-rotator galaxies with multiple tracer populations.

## Dependencies

* python >= 3.5
  * numpy
  * scipy
  * astropy
  * emcee
  * dill
  * multiprocess
  * ruamel.yaml
  * h5py
  * psutil

```
usage: slomo [-h] [--verbose] {init,sample,run,mock} ...

Construct and run dynamical models.

positional arguments:
  {init,sample,run,mock}
                        Available commands
    init                Construct a model
    sample              Sample from an existing model.
    run                 Shortcut for init and sample for new model output.
    mock                Create mock data from specified model parameters.

optional arguments:
  -h, --help            show this help message and exit
```

```
usage: slomo init [-h] [--clobber] config

positional arguments:
  config      Config file in YAML format. See docs for required entries.

optional arguments:
  -h, --help  show this help message and exit
  --clobber   If selected, overwrite existing hdf5 output file.
```

```
usage: slomo sample [-h] [--mock] [--threads THREADS] hdf5 niter

positional arguments:
  hdf5               hdf5 output file
  niter              Number of iterations to run.

optional arguments:
  -h, --help         show this help message and exit
  --mock             If selected, sample from mock data instead of stored
                     data.
  --threads THREADS  Number of threads to use.
```

```
usage: slomo run [-h] [--threads THREADS] config niter

positional arguments:
  config             Config file in YAML format. See docs for required
                     entries.
  niter              Number of iterations to run.

optional arguments:
  -h, --help         show this help message and exit
  --threads THREADS  Number of threads to use.
```
