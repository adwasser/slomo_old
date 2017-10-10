Running
=======

``slomo`` uses a standard UNIX-style command line interface for generating model files from model input files and sampling from the model.

You can access a minimal reminder of available subcommands with ``slomo -h``, and order of arguments for a subcommand with ``slomo subcommand -h``.

.. code-block:: none

    --------------------------------------------------
    slomo
    version : cffa80a48444cc6a8c865a8f935e18bce058a39c
    --------------------------------------------------
    Construct and run dynamical models.
    8 cpus available

    positional arguments:
      {init,info,sample,run}
                            Available commands
        init                Construct a model
        info                Quick info on the model.
        sample              Sample from an existing model.
        run                 Shortcut for init and sample for new model output.

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


info
----

This subcommand gives detailed information about the model hdf5 file.

.. code-block:: none

    usage: slomo info [-h] hdf5

    positional arguments:
      hdf5        hdf5 output file

    optional arguments:
      -h, --help  show this help message and exit

      
sample
------

This subcommand will draw ``niter`` samples and save them in the specified ``hdf5`` file.

.. code-block:: none

    usage: slomo sample [-h] [--threads THREADS] hdf5 niter

    positional arguments:
      hdf5               hdf5 output file
      niter              Number of iterations to run.

    optional arguments:
      -h, --help         show this help message and exit
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

