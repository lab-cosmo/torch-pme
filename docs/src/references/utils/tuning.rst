Tuning
######

The choice of parameters like the neighborlist ``cutoff``, the ``smearing`` or the
``lr_wavelength``/``mesh_spacing`` has a large influence one the accuracy of the
calculation. To help find the parameters that meet the accuracy requirements, this
module offers tuning methods for the calculators.

The scheme behind all tuning functions is a gradient-based optimization, which tries to
find the minimal of the error estimation formula and stops after the error is smaller
than the given accuracy. Because these methods are gradient-based, be sure to pay
attention to the ``learning_rate`` and ``max_steps`` parameter. A good choice of these
two parameters can enhance the optimization speed and performance.

.. autoclass:: torchpme.utils.tune_ewald
    :members:

.. autoclass:: torchpme.utils.tune_pme
    :members:
