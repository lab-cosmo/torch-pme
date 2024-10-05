.. _base_class:

Core classes
############

The core functionality of ``torch-pme`` entails calculators that 
can evaluate range-separated models of non-bonded interactions. 
The base classes that provide the interface to compute these types
of interactions are *calculators*, that take information on the 
periodic cell and the atom types and positions and computes
interatomic potential-like terms.

``torch-pme`` provides an interface in which the positions of the
atoms in a structure are stored in :py:class:`torch.Tensor` objects,
or in a :py:class:`metatensor.System <metatensor.torch.atomistic.System>`
object.

.. automodule:: torchpme.calculators.base
    :members:
    :undoc-members:
    :private-members: _compute_single_system

.. automodule:: torchpme.metatensor.base
    :members:
    :undoc-members:
