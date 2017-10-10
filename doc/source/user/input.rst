Input files
===========

.. _input-label:

YAML file format
################

``slomo`` uses `YAML <http://www.yaml.org/start.html>`_ as a markup language for describing model configurations.

Input files are parsed with :py:meth:`~slomo.io.read_yaml`, which returns a :py:class:`~slomo.models.DynamicalModel` instance.  The main keys of the file are listed below.  

params
------

These are the free parameters of the model.  The entry for `params` should be a list of dictionaries.  Each parameter dictionary should have an entry for `name`, `value`, and `lnprior`.  The chosen `name` should match the relevant keyword of any function which will need to be called during the likelihood call.  The `value` is simply the starting value. The `lnprior` value should be a tuple (comma separated, no surrounding parenthesis) of the prior name (taken from :py:mod:`~slomo.pdf`) and any numeric values for parameters of the distribution.  You can optionally include a `transform` key whose value is the name of a function from :py:mod:`~slomo.transforms`.

In the example below, `rho_s` has a uniform prior on the log (base 10) of the value.  Since any physics functions involving :math:`\rho_s` expect the actual value and not the log of the value, we specify that we want to use `transform.from_log` on the value.

.. code-block:: yaml

    params:
      - name: rho_s
        value: 6.84
        transform: from_log
        lnprior: lnuniform, 5, 9
      - name: r_s
        value: 1.74
        lnprior: lnuniform, 1, 3
        transform: from_log
      - name: gamma
        value: 0.6
        lnprior: lnuniform, 0, 2


constants
---------

These are any values needed for numeric evaluation of the likelihood but which are held constant.  The entry for `constants` should be a dictionary of key/value pairs, where the name of the key should match the keyword needed in any function calls.

.. code-block:: yaml

   constants:
       I0_b: 1
       Re_b: 345.61
       n_b: 1.6
       I0_r: 1
       Re_r: 169.13
       n_r: 1.6


mass_model
----------

This entry specifies the components of mass to be constrained.  This should be a list of keys whose values are functions from :py:mod:`~slomo.mass`.  The keys are not terribly important, but can be referenced later if you so desire.

.. code-block:: yaml

    mass_model:
      - dm: M_gNFW
      - st: L_sersic_s
      - bh: heaviside_bh

tracers
-------

Here we specify the kinematic tracers in our potential.  This entry is a list of dictionaries, where each dictionary should have a `name`, an `anisotropy`, a `surface_density`, and a `volume_density`.  The `name` will be referenced by the `measurements` dictionary, but is otherwise not important.  The remaining keys should be references to functions in :py:mod:`~slomo.anisotropy`, :py:mod:`~slomo.surface_density`, and :py:mod:`~slomo.volume_density` respectively.

.. code-block:: yaml

    tracers:
      - name: stars
        anisotropy: K_constant_s
        surface_density: I_sersic_s
        volume_density: nu_sersic_s
      - name: blue_gc
        anisotropy: K_constant_b
        surface_density: I_sersic_b
        volume_density: nu_sersic_b
      - name: red_gc
        anisotropy: K_constant_r
        surface_density: I_sersic_r
        volume_density: nu_sersic_r
    		

measurements
------------

This entry specifies the joint likelihood model and any constraining data.  This is a list of dictionaries, each of which should contain a keys of `name`, `likelihood`, `model`, `observables`, and `weight`.  The `name` will be saved and can be referred to later.  The `likelihood` should come from :py:mod:`~slomo.likelihood`.  The `model` should be a reference to a one of the defined tracers.  This can also be a list of tracers in the case of the :py:meth:`~slomo.likelihood.lnlike_gmm` likelihood.  The `observables` should be a filename as described below.  The `weight` value should be a boolean that specifies whether or not to use a hyperparameter to describe the trust in the dataset (following Ma & Berndsen, 2014).


Other keys
----------

These are miscellaneous other setting to pass to ``slomo``.  For instance, `nwalkers: 128` will tell ``slomo`` to use 128 walkers for sampling the posterior distribution.


Data file format
################

The data files are whitespace delimited.  They should have a header (first comment with ``#``) which specifies the variable names.  There should be a radius variable (notated as ``R``) for spatially-varying quantities.  For other variables, see the :py:mod:`~slomo.likelihood` call associated with the observable for the necessary keywords.  For instance, :py:meth:`~slomo.likelihood.lnlike_discrete` expects measurements of ``v`` and ``dv`` for a velocity and velocity uncertainty at each radial measurement.  :py:meth:`~slomo.likelihood.lnlike_continuous` expects measurements of ``sigma`` and ``dsigma`` for the velocity dispersion and associated uncertainty, so the data file might look like

::

   # R sigma dsigma
   3.080 266.426 6.179
   3.200 271.820 8.300
   3.394 263.766 6.199

