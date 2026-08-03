.. _prefactors:

Prefactors
==========

This module contains common ``prefactors`` for converting between Gaussian units and
other unit systems that can be used to set a unit system when initializing a
:ref:`calculator <calculators>`.

Many scientific calculators and computational tools, such as ``torchpme``, are set to
operate in Gaussian units by default, where the term :math:`1/(4 \pi \epsilon_0) = 1.0`,
where :math:`\epsilon_0` is the vacuum permittivity. When converting from Gaussian to SI
units, it is necessary to multiply by :math:`e^2/(4 \pi \epsilon_0)`, where :math:`e` is
the elementary charge.

To perform conversions from SI units to other unit systems, you can multiply by the
appropriate units of energy and divide by the appropriate units of length.

.. automodule:: torchpme.prefactors
    :members:
    :undoc-members:
