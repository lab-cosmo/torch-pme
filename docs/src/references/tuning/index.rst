Tuning
######

The choice of parameters like the neighborlist ``cutoff``, the ``smearing`` or the
``lr_wavelength``/``mesh_spacing`` has a large influence one the accuracy of the
calculation. To help find the parameters that meet the accuracy requirements, this
module offers tuning methods for the calculators.

For usual tuning procedures we provide simple functions like
:func:`torchpme.tuning.tune_ewald` that returns for a given system the optimal
parameters for the Ewald summation. For more complex tuning procedures, we provide
classes like :class:`torchpme.tuning.ewald.EwaldErrorBounds` that can be used to
implement custom tuning procedures.

.. important::

   Current tuning methods are only implemented for the Coulomb potential.

.. toctree::
   :maxdepth: 1
   :glob:

   ./*
