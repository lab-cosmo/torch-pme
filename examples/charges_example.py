"""
Computations with Multiple Charge Channels
==========================================
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
*NaCl* crystal, where we will cover both the Torch and Metatensor interfaces of
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
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborListOptions, System
from vesin import NeighborList

import meshlode


# %%
# Define a global constant for the cutoff of the neighbor list calculations.
CUTOFF = 1.0

# %%
# Create the properties NaCl unit cell
symbols = ("Na", "Cl")
types = torch.tensor([11, 17])
positions = torch.tensor([(0, 0, 0), (0.5, 0.5, 0.5)])
cell = torch.eye(3)

# %%
# Based on the system we compute the corresponding full neighbor list using `vesin
# <https://luthaf.fr/vesin>`_ and rearrange the results to be suitable for the
# calculations below.

nl = NeighborList(cutoff=CUTOFF, full_list=True)

i, j, S, D = nl.compute(points=positions, box=cell, periodic=True, quantities="ijSD")

i = torch.from_numpy(i.astype(int))
j = torch.from_numpy(j.astype(int))
neighbor_indices = torch.vstack([i, j])
neighbor_shifts = torch.from_numpy(S.astype(int))

distances = torch.from_numpy(D).reshape(-1, 3, 1)
distances = distances.type(positions.dtype)

# %%
# Next, we initialize the :py:class:`PMEPotential` calculator with an ``exponent`` of
# *1* for electrostatic interactions between the two atoms. This calculator
# will be used to *compute* the potential energy of the system.

calculator = meshlode.PMEPotential(exponent=1.0)

# %%
# Single Charge Channel
# #####################
# As a first application of multiple charge channels, we start simply by using the
# classic definition of one charge channel as a reshaped 1D array.

charges = torch.tensor([1.0, -1.0]).reshape(-1, 1)

# %%
# We reshaped the ``charges`` into a 2D array to be a suitable input to the calculator
# where the *rows* describe the number of atoms ``(2)`` and the *columns* the number of
# atomic charge channels ``(1)``.

charges.shape

# %%
# Calculate the potential using the PMEPotential calculator

potential = calculator(
    positions=positions,
    cell=cell,
    charges=charges,
    neighbor_indices=neighbor_indices,
    neighbor_shifts=neighbor_shifts,
)

# %%
# We find a potential that is close to the Madelung constant of a NaCl crystal which is
# :math:`2 \cdot 1.7626 / \sqrt{3} \approx 2.0354`.

print(potential)

# %%
# Species-wise One-Hot Encoded Charges
# ####################################
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

charges_one_hot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

# %%
# While in ``charges`` there was only one row, ``charges_one_hot`` contains two rows
# where the first one corresponds to the Na channel and the second one to the Cl
# channel. Consequently, the charge of the Na atom in the Cl channel at index ``(0,1)``
# of the ``charges_one_hot`` is zero as well as the ``(1,0)`` which corresponds to the
# charge of Cl in the Na channel.
#
# We now again calculate the potential using the same ``PMEPotential`` calculator using
# the ``charges_one_hot`` as input.

potential_one_hot = calculator(
    positions=positions,
    cell=cell,
    charges=charges_one_hot,
    neighbor_indices=neighbor_indices,
    neighbor_shifts=neighbor_shifts,
)

# %%
# Note that the potential has the same shape as the input charges, but there is a finite
# potential on the position of the Cl in the Na channel.

print(potential_one_hot)

# %%
# From the potential we can recover the Madelung as above by summing the charge channel
# contribution multiplying by the actual partial charge of the atoms.

charge_Na = 1.0
charge_Cl = -1.0
print(charge_Na * potential_one_hot[0] + charge_Cl * potential_one_hot[1])

# %%
# Metatensor Version
# ------------------
# Next, we will perform the same exercise with the Metatensor interface. This involves
# creating a new calculator with the metatensor interface.

calculator_metatensor = meshlode.metatensor.PMEPotential(exponent=1.0)

# %%
# Computation with metatensor involves using Metatensor's :py:class:`System
# <metatensor.torch.atomistic.System>` class. The ``System`` stores atomic ``types``,
# ``positions``, and ``cell`` dimensions.
#
# .. note::
#    For our calculations, the parameter ``types`` passed to a ``System`` is redundant;
#    it will not be directly used to calculate the potential as the potential depends
#    only on the charge of the atom, NOT on the atom's type. However, we still have to
#    pass them because it is an obligatory parameter to build the `System` class.

system = System(types=types, positions=positions, cell=cell)

# %%
# We first add the neighbor list to the ``system``. This requires creating a
# ``NeighborList`` object to store the *neighbor indices*, *distances*, and *shifts*.
# The :py:class:`NeighborListOptions <metatensor.torch.atomistic.NeighborListOptions>`
# class is used to specify the ``cutoff`` distance and whether to use the ``full``
# neighbor list.

sample_values = torch.hstack([neighbor_indices.T, neighbor_shifts])
samples = Labels(
    names=[
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ],
    values=sample_values,
)

components = Labels(names=["xyz"], values=torch.tensor([[0, 1, 2]]).T)
properties = Labels(names=["distance"], values=torch.tensor([[0]]))
neighbor_list = TensorBlock(
    torch.tensor(distances).view(-1, 3, 1), samples, [components], properties
)

nl_options = NeighborListOptions(cutoff=CUTOFF, full_list=True)
system.add_neighbor_list(options=nl_options, neighbors=neighbor_list)

# %%
# Now the ``system`` is ready to be used inside the calculators
#
# Single Charge Channel
# #####################
# For the metatensor branch, charges of the atoms are defined in a tensor format and
# attached to the system as a :py:class:`TensorBlock <metatensor.torch.TensorBlock>`.

# Create a TensorBlock for the charges
data = TensorBlock(
    values=charges,
    samples=Labels.range("atom", charges.shape[0]),
    components=[],
    properties=Labels.range("charge", charges.shape[1]),
)

# %%
# Add the charges data to the system

system.add_data(name="charges", data=data)

# %%
# We now calculate the potential using the MetaPMEPotential calculator

potential_metatensor = calculator_metatensor.compute(system)

# %%
# The calculated potential is wrapped inside a :py:class:`TensorMap
# <metatensor.torch.TensorMap>` and annotated with metadata of the computation.

print(potential_metatensor)

# %%
# The tensorMap has *1* :py:class:`TensorBlock <metatensor.torch.TensorBlock>` and the
# values of the potential are stored in the ``values`` property.

print(potential_metatensor[0].values)

# %%
# The ``values`` are the same results as for the torch interface shown above
# The metadata associated with the ``TensorBlock`` tells us that we have 2 ``samples``
# which are our two atoms and 1 property which corresponds to one charge channel.

print(potential_metatensor[0])

# %%
# If you want to inspect the metadata in more detail, you can access the
# :py:class:`Labels <metatensor.torch.Labels>` using the
# ``potential_metatensor[0].properties`` and ``potential_metatensor[0].samples``
# attributes.
#
# Species-wise One-Hot Encoded Charges
# ####################################
# We now create new charges data based on the species-wise ``charges_one_hot`` and
# overwrite the ``system``'s charges data using ``override=True`` when applying the
# :py:meth:`add_data <metatensor.torch.atomistic.System.add_data>` method.

data_one_hot = TensorBlock(
    values=charges_one_hot,
    samples=Labels.range("atom", charges_one_hot.shape[0]),
    components=[],
    properties=Labels.range("charge", charges_one_hot.shape[1]),
)

# Add the charges data to the system
system.add_data(name="charges", data=data_one_hot, override=True)

# %%
# Finally, we calculate the potential using ``calculator_metatensor``
potential = calculator_metatensor.compute(system)

# %%
# And as above, the values of the potential are the same.

print(potential[0].values)
