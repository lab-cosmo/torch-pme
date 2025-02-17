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

Fixed
#####


* Fix inconsistent ``cutoff`` in neighbor list example
=======
* Fixed consistency of dtype and device in the SplinePotential class
* All calculators now check if the cell is zero if the potential is range-separated


`Version 0.1.0 <https://github.com/lab-cosmo/torch-pme/releases/tag/v0.1.0>`_ - 2024-12-05
------------------------------------------------------------------------------------------

Added
#####

* First release outside of the lab
