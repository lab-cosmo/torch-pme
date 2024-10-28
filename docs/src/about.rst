What is torch-pme
=================

``torch-pme`` provides an interface in which the positions of the atoms in a structure
are stored in :py:class:`torch.Tensor` objects, or in a :py:class:`metatensor.System
<metatensor.torch.atomistic.System>` object.

torch-pme can be used as qnend-to-end library compuating the potential from
positions/charged and as a modular library to constract complex fourier space workflows.

For using ``torch-pme`` as an end-to-end library the main entry point are *calculators*
which take :ref:`potentials` instances as input and return the calculated potential as
output.
