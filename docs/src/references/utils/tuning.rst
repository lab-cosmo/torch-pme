Tuning
######

The choice of parameters like the neighborlist ``cutoff``, the ``smearing`` or the
``lr_wavelength``/``mesh_spacing`` greatly influence the accuracy of the calculation
results. To help find the parameters that meet the accuracy requirements, this module
offers tuning methods for the :py:class:`torchpme.EwaldCalculator` and the
:py:class:`torchpme.PMECalculator`.

The scheme behind these tuning methods is gradient-based optimization, which tries to
find the minimal of the error estimation formula and stops after the error is smaller
than the given accuracy. Because these methods are gradient-based, be sure to pay
attention to the ``learning_rate`` and ``max_steps`` parameter. A good choice of these
two parameters can enhance the optimization speed and performance.

.. automodule:: torchpme.utils.tuning
    :members:
    :undoc-members:
