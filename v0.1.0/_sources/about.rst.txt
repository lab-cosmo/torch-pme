What is torch-pme
=================

``torch-pme`` provides an interface in which the positions of the atoms in a structure
are stored in :class:`torch.Tensor` objects, or in a :class:`metatensor.System
<metatensor.torch.atomistic.System>` object.

torch-pme can be used as an end-to-end library computing the potential from
positions/charged and as a modular library to construct complex Fourier-domain
architectures.

To use ``torch-pme`` as an end-to-end library the main entry point are
:ref:`calculators`, that compute pair :ref:`potentials` combining real-space and k-space
components. They take a description of a structure as input and return the calculated
potential at the atomic positions as output.
