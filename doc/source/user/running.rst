Running
=======

``slomo`` uses a standard UNIX-style command line interface for generating model files from model input files and sampling from the model.

You can access a minimal reminder of available subcommands with ``slomo -h``, and order of arguments for a subcommand with ``slomo subcommand -h``.

.. code-block:: none
   
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

init
----

This subcommand constructs a model file (see :ref:`output-label`) from a model configuation file (see :ref:`input-label`).

.. code-block:: none
   
   usage: slomo init [-h] [--clobber] config

   positional arguments:
   config      Config file in YAML format. See docs for required entries.

   optional arguments:
   -h, --help  show this help message and exit
   --clobber   If selected, overwrite existing hdf5 output file.

sample
------

This subcommand will draw ``niter`` samples and save them in the specified ``hdf5`` file.

.. code-block:: none
   
   usage: slomo sample [-h] [--mock] [--threads THREADS] hdf5 niter

   positional arguments:
   hdf5               hdf5 output file
   niter              Number of iterations to run.

   optional arguments:
   -h, --help         show this help message and exit
   --mock             If selected, sample from mock data instead of stored data.
   --threads THREADS  Number of threads to use.

run
---

This is a shortcut to both call ``init`` and start sampling from a specified `config` file.

.. code-block:: none
   
   usage: slomo run [-h] [--threads THREADS] config niter

   positional arguments:
   config             Config file in YAML format. See docs for required
   entries.
   niter              Number of iterations to run.

   optional arguments:
   -h, --help         show this help message and exit
   --threads THREADS  Number of threads to use.

