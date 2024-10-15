.. _prefactors:

Prefactors
==========

This file contains the prefactors required for converting between different unit
systems.

- Many scientific calculators and computational tools, such as torchpme, are
  designed to work in Gaussian units by default, where :math:`\frac{1}{4 \pi
  \epsilon_0}` is 1.
- To convert from Gaussian units to SI units, we need to multiply by
  :math:`\frac{e^2}{4 \pi \epsilon_0}`.
- To convert from SI to other units, one can multiply by the appropriate units
  of energy and divide by the appropriate units of length.
- Below are the prefactors for converting between Gaussian units and other
  units.
- More prefactors can be added as needed in the ``torchpme.utils.prefactors``
  module.

.. automodule:: torchpme.utils.prefactors
    :members:
    :undoc-members:
