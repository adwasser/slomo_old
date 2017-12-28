Installing
==========

Dependencies
------------

``slomo`` uses python3; in particular, it needs a version >= 3.5.

Other dependencies listed below should be installed when installing with ``pip`` or through `setup.py`.

* numpy
* scipy
* astropy
* emcee
* dill
* multiprocess
* ruamel.yaml
* h5py
* psutil


pip
---

Using ``pip``:

.. code-block:: none

   pip install git+git://github.com/adwasser/slomo.git   


Git
---

Using ``git``:

.. code-block:: none
		
   git clone https://github.com/adwasser/slomo.git
   cd slomo
   python setup.py develop
