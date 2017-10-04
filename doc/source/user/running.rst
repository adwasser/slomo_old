Running
=======

``slomo`` uses a standard UNIX-style command line interface for generating model files from model input files and sampling from the model.

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


.. code-block:: none
   
   usage: slomo init [-h] [--clobber] config

   positional arguments:
   config      Config file in YAML format. See docs for required entries.

   optional arguments:
   -h, --help  show this help message and exit
   --clobber   If selected, overwrite existing hdf5 output file.


.. code-block:: none
   
   usage: slomo sample [-h] [--mock] [--threads THREADS] hdf5 niter

   positional arguments:
   hdf5               hdf5 output file
   niter              Number of iterations to run.

   optional arguments:
   -h, --help         show this help message and exit
   --mock             If selected, sample from mock data instead of stored
   data.
   --threads THREADS  Number of threads to use.


.. code-block:: none
   
   usage: slomo run [-h] [--threads THREADS] config niter

   positional arguments:
   config             Config file in YAML format. See docs for required
   entries.
   niter              Number of iterations to run.

   optional arguments:
   -h, --help         show this help message and exit
   --threads THREADS  Number of threads to use.

