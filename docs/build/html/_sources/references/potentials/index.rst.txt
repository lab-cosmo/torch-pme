.. _potentials:

Potentials
###########

The :class:`torchpme.Potential` class defines an API for general pair potentials, that
implement methods computing the different terms needed for a range-separated calculator.
In other terms, a full implementation of the API computes short-range and long-range
potentials as a function of the interatomic distance, as well as the radial Fourier
transform of the long-range contribution, that is needed to compute the potential with
k-space methods.

The most common potentials are the :class:`torchpme.CoulombPotential`` and the
:class:`torchpme.InversePowerLawPotential` but ``torch-pme`` also provides additional
potentials for more specialized purposes, or to implement long-range architectures that
go beyond physics-based modeling.

Implemented Potentials
----------------------

.. toctree::
   :maxdepth: 1
   :glob:

   ./*
