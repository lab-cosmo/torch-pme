.. _userdoc-changelog:

Changelog
=========

All notable changes to ``torch-pme`` are documented here, following the `keep a
changelog <https://keepachangelog.com/en/1.1.0/>`_ format. This project follows
`Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

.. Possible sections for each release:

.. Added
.. #####

.. Fixed
.. #####

.. Changed
.. #######

.. Removed
.. #######

`Unreleased <https://github.com/lab-cosmo/torch-pme/>`_
-------------------------------------------------------

Added
#####

* Updated to ``metatensor-torch`` version 0.7
* Added support for Python 3.13
* Added classes for the calculation of dipole interactions
* Better documentation for for ``cell``, ``charges`` and ``positions`` parameters

Removed
#######

* Remove ``device`` and ``dtype`` from init of ``Calculator``, ``Potential`` and
  ``Tuning`` classes

`Version 0.2.0 <https://github.com/lab-cosmo/torch-pme/releases/tag/v0.2.0>`_ - 2025-01-23
------------------------------------------------------------------------------------------

Added
#####

* Added a PyTorch implementation of the exponential integral function
* Added ``dtype`` and ``device`` for ``Calculator`` classes
* Added an example on the tuning scheme and usage, and how to optimize the ``cutoff``

Changed
#######

* Removed ``utils`` module. ``utils.tuning`` and ``utils.prefactor`` are now in the root
  of the package; ``utils.splines`` is now in the ``lib`` module
* Tuning now uses a grid-search based scheme, instead of a gradient based scheme
* Tuning functions no longer takes the ``cutoff`` parameter, and thus does not
  support a built-in NL calculation.
* Refactor the ``InversePowerLawPotential`` class to restrict the exponent to integer
  values

Fixed
#####

* Ensured consistency of ``dtype`` and ``device`` in the ``Potential`` and
  ``Calculator`` classses
* Fixed consistency of ``dtype`` and ``device`` in the ``SplinePotential`` class
* Fix inconsistent ``cutoff`` in neighbor list example
* All calculators now check if the cell is zero if the potential is range-separated

`Version 0.1.0 <https://github.com/lab-cosmo/torch-pme/releases/tag/v0.1.0>`_ - 2024-12-05
------------------------------------------------------------------------------------------

Added
#####

* First release outside of the lab
