# %% [markdown]
# # Computations with Explicit Charges using Meshlode

# %% [markdown]
# In this tutorial, we will demonstrate how to perform
# computations with explicit charges using the Meshlode library. We will cover
# both the Torch and Metatensor branches of Meshlode.

# %% [markdown]
# ## Torch Version
#
# First, we will work with the Torch version of Meshlode. This involves using
# PyTorch for tensor operations and ASE (Atomic Simulation Environment) for
# creating and manipulating atomic structures.

# %%
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborListOptions, System

from meshlode import PMEPotential
from meshlode.metatensor import PMEPotential as MetaPMEPotential


# %% [markdown]
# ### Setting up the NaCl Unit Cell

# %%
# Define constants
LATTICE_CONSTANT = 1.0
CUTOFF = 1.0

# Create the NaCl unit cell
nacl = Atoms(
    symbols=["Na", "Cl"],
    positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
    cell=torch.eye(3) * LATTICE_CONSTANT,
    pbc=True,
)

# Create the NeighborList object
atom_is, atom_js, neighbor_shifts = neighbor_list(
    "ijS", nacl, CUTOFF, self_interaction=False
)

# %% [markdown]
# ### Initializing the PMEPotential Calculator

# %%
# Initialize the PMEPotential calculator with a specific exponent and short-range cutoff
calculator = PMEPotential(
    exponent=1.0,
    sr_cutoff=CUTOFF,
)

# %% [markdown]
# The `PMEPotential` calculator is initialized with an exponent
# and a short-range cutoff. This calculator will be used to compute the
# potential energy of the system.

# %% [markdown]
# ### Computing the Potential with Single Channel Charges

# %%
# Define the charges of the atoms as a 2D tensor, where each column
# represents an atom's charge
charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)

# Calculate the potential using the PMEPotential calculator
potential = calculator(
    positions=torch.tensor(
        nacl.positions, dtype=torch.float64
    ),  # Positions of the atoms in the NaCl structure
    cell=torch.tensor(
        nacl.cell, dtype=torch.float64
    ),  # Simulation cell dimensions for NaCl
    charges=charges,  # Charges of the atoms
    neighbor_indices=[
        torch.stack([torch.tensor(atom_is), torch.tensor(atom_js)])
    ],  # Indices of neighboring atoms
    neighbor_shifts=torch.tensor(
        neighbor_shifts
    ),  # Shifts for periodic boundary conditions
)

# Print the calculated potential
print(potential)

# %% [markdown]
# The charges of the atoms are defined in a tensor format and
# passed to the `PMEPotential` calculator. The calculated potential is then
# printed.

# %% [markdown]
# ### Computing the Potential with Multiple Channels for Charges

# %%
# One can also use several channels for charges, e.g., for a system with 2
# channels and 2 atoms as one-hot encoding
charges = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

# Calculate the potential using the PMEPotential calculator
potential = calculator(
    positions=torch.tensor(
        nacl.positions, dtype=torch.float64
    ),  # Positions of the atoms in the NaCl structure
    cell=torch.tensor(
        nacl.cell, dtype=torch.float64
    ),  # Simulation cell dimensions for NaCl
    charges=charges,  # Charges of the atoms
    neighbor_indices=[
        torch.stack([torch.tensor(atom_is), torch.tensor(atom_js)])
    ],  # Indices of neighboring atoms
    neighbor_shifts=torch.tensor(
        neighbor_shifts
    ),  # Shifts for periodic boundary conditions
)

# Print the calculated potential
print(potential)

# %% [markdown]
# We can also define charges with multiple channels, which allows
# for more complex charge distributions. The calculated potential for this setup
# is then printed.

# %% [markdown]
# ## Metatensor Version

# %% [markdown]
# Next, we will work with the Metatensor version of Meshlode. This
# involves using Metatensor for tensor operations while still utilizing ASE for
# atomic structure creation.

# %% [markdown]
# ### Setting up the NaCl Unit Cell

# %%
# Define constants
LATTICE_CONSTANT = 1.0
CUTOFF = 1.0

# Create the NaCl unit cell
nacl = Atoms(
    symbols=["Na", "Cl"],
    positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
    cell=torch.eye(3) * LATTICE_CONSTANT,
    pbc=True,
)

# Create the NeighborList object
atom_is, atom_js, distances, neighbor_shifts = neighbor_list(
    "ijDS", nacl, CUTOFF, self_interaction=False
)

# %% [markdown]
# ### Initializing the PMEPotential Calculator

# %%
# Initialize the PMEPotential calculator with a specific exponent and short-range cutoff
calculator = MetaPMEPotential(
    exponent=1.0,
    sr_cutoff=CUTOFF,
)

# %% [markdown]
# The `MetaPMEPotential` calculator is initialized with an
# exponent and a short-range cutoff, similar to the Torch version.

# %% [markdown]
# ### Setting Up the System and Adding Charges

# %%
# Define the system with atomic types, positions, and cell dimensions
system = System(
    types=torch.tensor(nacl.numbers),
    positions=torch.tensor(nacl.positions, dtype=torch.float64),
    cell=torch.tensor(nacl.cell, dtype=torch.float64),
)

samples = Labels(
    names=["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
    values=torch.stack(
        [
            torch.tensor(atom_is),
            torch.tensor(atom_js),
            torch.tensor(neighbor_shifts[:, 0]),
            torch.tensor(neighbor_shifts[:, 1]),
            torch.tensor(neighbor_shifts[:, 2]),
        ],
        axis=1,
    ),
)

components = Labels(names=["xyz"], values=torch.tensor([[0, 1, 2]]).T)

properties = Labels(names=["distance"], values=torch.tensor([[0]]))

neighbor_list = TensorBlock(
    torch.tensor(distances).view(-1, 3, 1), samples, [components], properties
)

nl_options = NeighborListOptions(cutoff=CUTOFF, full_list=True)

# %% [markdown]
# The system is defined using Metatensor's `System` class, which
# includes atomic types, positions, and cell dimensions. We also create a
# `NeighborList` object to store the neighbor indices, distances, and shifts.
# The `NeighborListOptions` class is used to specify the cutoff distance and
# whether to use the full neighbor list.
#
# NOTE: For our calculations, the parameter `types` passed to a `System` is
# redundant; it will not be directly used to calculate the potential as the
# potential depends only on the charge of the atom, not on the atom's type.
# However, we still have to pass them because it is an obligatory parameter to
# build the `System` class.

# %% [markdown]
# ### Computing the Potential with Single Channel Charges

# %%
# Define the charges of the atoms
charges = torch.tensor([[1.0], [-1.0]], dtype=torch.double)

# Create a TensorBlock for the charges
data = TensorBlock(
    values=charges,
    samples=Labels.range("atom", charges.shape[0]),
    components=[],
    properties=Labels.range("charge", charges.shape[1]),
)

# Add the charges data to the system
system.add_data(name="charges", data=data)
system.add_neighbor_list(options=nl_options, neighbors=neighbor_list)

# Calculate the potential using the MetaPMEPotential calculator
potential = calculator.compute(system)

# Print the calculated potential
print(potential[0].values)

# %% [markdown]
# The charges of the atoms are defined in a tensor format and
# added to the system as a `TensorBlock`. The calculated potential is then
# printed.

# %% [markdown]
# ### Computing the Potential with Multiple Channels for Charges

# %%
# Define the system again to reset
system = System(
    types=torch.tensor(nacl.numbers),
    positions=torch.tensor(nacl.positions, dtype=torch.double),
    cell=torch.tensor(nacl.cell, dtype=torch.double),
)
system.add_neighbor_list(options=nl_options, neighbors=neighbor_list)

# Define charges with multiple channels (e.g., one-hot encoding)
charges = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.double)

# Create a TensorBlock for the charges
data = TensorBlock(
    values=charges,
    samples=Labels.range("atom", charges.shape[0]),
    components=[],
    properties=Labels.range("charge", charges.shape[1]),
)

# Add the charges data to the system
system.add_data(name="charges", data=data)

# Calculate the potential using the MetaPMEPotential calculator
potential = calculator.compute(system)

# Print the calculated potential
print(potential[0].values)

# %% [markdown]
# We can also define charges with multiple channels and add them to the system.
# %%
