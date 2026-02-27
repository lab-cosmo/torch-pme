"""
Computations with Multiple Charge Channels
==========================================

.. currentmodule:: torchpme

In a physical system, the (electrical) charge is a scalar atomic property, and besides
the distance between the particles, the charge defines the electrostatic potential. When
computing a potential with Meshlode, you can not only pass a (reshaped) 1-D array
containing the charges to the ``compute`` method of calculators, but you can also pass a
2-D array containing multiple charges per atom. Meshlode will then calculate one
potential per so-called *charge-channel*. For the standard electrostatic potential, the
number of charge channels is *1*. Additional charge channels are especially useful in a
machine learning task where, for example, one wants to create one potential for each
species in an atomistic system using a so-called *one-hot encoding* of charges.

Here, we will demonstrate how to use the ability of multiple charge channels for a
*CsCl* crystal, where we will cover both the Torch and Metatensor interfaces of
Meshlode.

Torch Version
--------------
First, we will work with the Torch version of Meshlode. This involves using `PyTorch`_
for tensor operations and `ASE`_ (Atomic Simulation Environment) for creating and
manipulating atomic structures.

.. _`PyTorch`: https://pytorch.org
.. _`ASE`: https://wiki.fysik.dtu.dk/ase
"""

# %%

import torch
import vesin.metatomic
import vesin.torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, System

import torchpme
from torchpme.tuning import tune_pme

dtype = torch.float64

# %%
#
# Create the properties CsCl unit cell
symbols = ("Cs", "Cl")
types = torch.tensor([55, 17])
charges = torch.tensor([[1.0], [-1.0]], dtype=dtype)
positions = torch.tensor([(0, 0, 0), (0.5, 0.5, 0.5)], dtype=dtype)
cell = torch.eye(3, dtype=dtype)
pbc = torch.tensor([True, True, True])


# %%
#
# Based on our system we will first *tune* the PME parameters for an accurate computation.
# The ``sum_squared_charges`` is equal to ``2.0`` becaue each atom either has a charge
# of 1 or -1 in units of elementary charges.

cutoff = 4.4
nl = vesin.torch.NeighborList(cutoff=cutoff, full_list=False)
neighbor_indices, neighbor_distances = nl.compute(
    points=positions.to(dtype=torch.float64, device="cpu"),
    box=cell.to(dtype=torch.float64, device="cpu"),
    periodic=True,
    quantities="Pd",
)
smearing, pme_params, _ = tune_pme(
    charges=charges,
    cell=cell,
    positions=positions,
    cutoff=cutoff,
    neighbor_indices=neighbor_indices,
    neighbor_distances=neighbor_distances,
)

# %%
#
# The tuning found the following best values for our system.

print("smearing:", smearing)
print("PME parameters:", pme_params)
print("cutoff:", cutoff)

# %%
#
# Based on the system we compute the corresponding half neighbor list using `vesin
# <https://luthaf.fr/vesin>`_ and rearrange the results to be suitable for the
# calculations below.

nl = vesin.torch.NeighborList(cutoff=cutoff, full_list=False)

neighbor_indices, S, D, neighbor_distances = nl.compute(
    points=positions, box=cell, periodic=True, quantities="PSDd"
)

# %%
#
# Next, we initialize the :class:`PMECalculator` calculator with an ``exponent`` of
# *1* for electrostatic interactions between the two atoms. This calculator
# will be used to *compute* the potential energy of the system.

calculator = torchpme.PMECalculator(
    torchpme.CoulombPotential(smearing=smearing), **pme_params
)
calculator.to(dtype=dtype)
# %%
#
# Single Charge Channel
# #####################
#
# As a first application of multiple charge channels, we start simply by using the
# classic definition of one charge channel per atom.

charges = torch.tensor([[1.0], [-1.0]], dtype=dtype)

# %%
#
# Any input the calculators has to be a 2D array where the *rows* describe the number of
# atoms (here ``(2)``) and the *columns* the number of atomic charge channels (here
# ``(1)``).

print(charges.shape)

# %%
#
# Calculate the potential using the PMECalculator calculator

potential = calculator(
    positions=positions,
    cell=cell,
    charges=charges,
    neighbor_indices=neighbor_indices,
    neighbor_distances=neighbor_distances,
)

# %%
#
# We find a potential that is close to the Madelung constant of a CsCl crystal which is
# :math:`2 \cdot 1.76267 / \sqrt{3} \approx 2.0354`.

print(charges.T @ potential)

# %%
#
# Species-wise One-Hot Encoded Charges
# ####################################
#
# Now we will compute the potential with multiple channels for the charges. We will use
# one channel per species and set the charges to *1* if the atomic ``symbol`` fits the
# correct channel. This is called one-hot encoding for each species.
#
# One-hot encoding is a powerful technique in machine learning where categorical data is
# converted into a binary vector representation. Each category is represented by a
# vector with all elements set to zero except for the index corresponding to that
# category, which is set to one. This allows the model to easily distinguish between
# different categories without imposing any ordinal relationship between them. In the
# context of molecular systems, one-hot encoding can be used to represent different
# atomic species as separate charge channels, enabling the calculation of
# species-specific potentials and facilitating the learning process for machine learning
# algorithms.

charges_one_hot = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)

# %%
#
# While in ``charges`` there was only one row, ``charges_one_hot`` contains two rows
# where the first one corresponds to the Na channel and the second one to the Cl
# channel. Consequently, the charge of the Na atom in the Cl channel at index ``(0,1)``
# of the ``charges_one_hot`` is zero as well as the ``(1,0)`` which corresponds to the
# charge of Cl in the Na channel.
#
# We now again calculate the potential using the same :class:`PMECalculator` calculator
# using the ``charges_one_hot`` as input.

potential_one_hot = calculator(
    charges=charges_one_hot,
    cell=cell,
    positions=positions,
    neighbor_indices=neighbor_indices,
    neighbor_distances=neighbor_distances,
)

# %%
#
# Note that the potential has the same shape as the input charges, but there is a finite
# potential on the position of the Cl in the Na channel.

print(potential_one_hot)

# %%
#
# From the potential we can recover the Madelung as above by summing the charge channel
# contribution multiplying by the actual partial charge of the atoms.

charge_Na = 1.0
charge_Cl = -1.0
print(charge_Na * potential_one_hot[0] + charge_Cl * potential_one_hot[1])

# %%
#
# Metatensor Version
# ------------------
# Next, we will perform the same exercise with the Metatensor interface. This involves
# creating a new calculator with the metatensor interface.

calculator_metatensor = torchpme.metatensor.PMECalculator(
    torchpme.CoulombPotential(smearing=smearing), **pme_params
)
calculator_metatensor.to(dtype=dtype)
# %%
#
# Computation with metatensor involves using Metatensor's :class:`System
# <metatomic.torch.System>` class. The ``System`` stores atomic ``types``,
# ``positions``, and ``cell`` dimensions.
#
# .. note::
#    For our calculations, the parameter ``types`` passed to a ``System`` is redundant;
#    it will not be directly used to calculate the potential as the potential depends
#    only on the charge of the atom, NOT on the atom's type. However, we still have to
#    pass them because it is an obligatory parameter to build the `System` class.

system = System(types=types, positions=positions, cell=cell, pbc=pbc)

# %%
#
# We now compute the neighborlist for our ``system`` using the `vesin metatensor
# interface <https://luthaf.fr/vesin/latest/metatensor.html>`_. This requires creating a
# :class:`NeighborListOptions <metatomic.torch.NeighborListOptions>` to set
# the cutoff and the type of list.

options = NeighborListOptions(cutoff=4.0, full_list=True, strict=False)
nl_mts = vesin.metatomic.NeighborList(options, length_unit="Angstrom")
neighbors = nl_mts.compute(system)

# %%
#
# Now the ``system`` is ready to be used inside the calculators
#
# Single Charge Channel
# #####################
#
# For the metatensor branch, charges of the atoms are defined in a tensor format and
# attached to the system as a :class:`TensorBlock <metatensor.torch.TensorBlock>`.
#
# Create a :class:`TensorMap <metetensor.torch.TensorMap>` for the charges

block = TensorBlock(
    values=charges,
    samples=Labels.range("atom", charges.shape[0]),
    components=[],
    properties=Labels.range("charge", charges.shape[1]),
)

tensor = TensorMap(
    keys=Labels("_", torch.zeros(1, 1, dtype=torch.int32)), blocks=[block]
)

# %%
#
# Add the charges data to the system

system.add_data(name="charges", tensor=tensor)

# %%
#
# We now calculate the potential using the MetaPMECalculator calculator

potential_metatensor = calculator_metatensor.forward(system, neighbors)

# %%
#
# The calculated potential is wrapped inside a :class:`TensorMap
# <metatensor.torch.TensorMap>` and annotated with metadata of the computation.

print(potential_metatensor)

# %%
#
# The tensorMap has *1* :class:`TensorBlock <metatensor.torch.TensorBlock>` and the
# values of the potential are stored in the ``values`` property.

print(potential_metatensor[0].values)

# %%
#
# The ``values`` are the same results as for the torch interface shown above
# The metadata associated with the ``TensorBlock`` tells us that we have 2 ``samples``
# which are our two atoms and 1 property which corresponds to one charge channel.

print(potential_metatensor[0])

# %%
#
# If you want to inspect the metadata in more detail, you can access the
# :class:`Labels <metatensor.torch.Labels>` using the
# ``potential_metatensor[0].properties`` and ``potential_metatensor[0].samples``
# attributes.
#
# Species-wise One-Hot Encoded Charges
# ####################################
#
# We now create new charges data based on the species-wise ``charges_one_hot`` and
# overwrite the ``system``'s charges data using ``override=True`` when applying the
# :meth:`add_data <metatomic.torch.System.add_data>` method.

block_one_hot = TensorBlock(
    values=charges_one_hot,
    samples=Labels.range("atom", charges_one_hot.shape[0]),
    components=[],
    properties=Labels.range("charge", charges_one_hot.shape[1]),
)

tensor_one_hot = TensorMap(
    keys=Labels("_", torch.zeros(1, 1, dtype=torch.int32)), blocks=[block_one_hot]
)

# %%
#
# Add the charges data to the system. We use the ``override=True`` to overwrite the
# already existing charge data from above.

system.add_data(name="charges", tensor=tensor_one_hot, override=True)

# %%
#
# Finally, we calculate the potential using ``calculator_metatensor``
potential = calculator_metatensor.forward(system, neighbors)

# %%
#
# And as above, the values of the potential are the same.

print(potential[0].values)

# %%
