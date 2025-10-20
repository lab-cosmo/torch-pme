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

* Add support for batched calculations


`Version 0.3.2 <https://github.com/lab-cosmo/torch-pme/releases/tag/v0.3.2>`_ - 2025-10-07
------------------------------------------------------------------------------------------

Added
#####

* Add support for slab geometries in Ewald and PME calculators

Fixed
#####

* Fix the error formula of Ewald used in tuning
* Fix wrong parameter checking

`Version 0.3.1 <https://github.com/lab-cosmo/torch-pme/releases/tag/v0.3.1>`_ - 2025-06-01
------------------------------------------------------------------------------------------

Added
#####

* Add a nan check for ``KSpaceFilter``
* Allow passing ``full_neighbor_list`` and ``prefactor`` to tuning functions
* Added backlink to the Cookbook
* Fix the lr_wavelength in the combined potential example to make the results coincide with the theoretical ones
* Update paper reference to JCP
* Introduce a new cutoff function for more aggressive exclusion.

Fixed
#####

* Fix gradient recording in Quickstart
* Update metatensor bindings to metatomic


Fixed
#####

* Fix exclusion_radius to work in a direct calculator

`Version 0.3.0 <https://github.com/lab-cosmo/torch-pme/releases/tag/v0.3.0>`_ - 2025-02-21
------------------------------------------------------------------------------------------

Added
#####

* Added support for Python 3.13
* Updated to ``metatensor-torch`` version 0.7
* Add a method to select a subset of a neighbor list based on a new ``cutoff``
  (:meth:`torchpme.tuning.tuner.TunerBase.filter_neighbors`)
* Added an *Ewald* calculator  for computing dipole-dipole interactions
  (:class:`torchpme.CalculatorDipole`) using a dipolar potential
  (:class:`torchpme.PotentialDipole`)
* Better documentation for for ``cell``, ``charges`` and ``positions`` parameters

Removed
#######

* Remove ``device`` and ``dtype`` from the init from all ``Calculator``, ``Potential``
  and ``Tuning`` classes

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
* Refactor the :class:`torchpme.InversePowerLawPotential`` class to restrict the
  exponent to integer values

Fixed
#####

* Ensured consistency of ``dtype`` and ``device`` in the ``Potential`` and
  ``Calculator`` classses
* Fixed consistency of ``dtype`` and ``device`` in the :class:`torchpme.SplinePotential`
  class
* Fix inconsistent ``cutoff`` in neighbor list example
* All calculators now check if the cell is zero if the potential is range-separated

`Version 0.1.0 <https://github.com/lab-cosmo/torch-pme/releases/tag/v0.1.0>`_ - 2024-12-05
------------------------------------------------------------------------------------------

Added
#####

* First release outside of the lab
