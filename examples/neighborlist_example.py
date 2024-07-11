"""
Computations with explicit Neighbor Lists
=========================================

This example will explain how to use the metatensor branch of Meshlode with an attached
neighborlist to an :py:class:`metatensor.torch.atomistic.System` object.
"""

# %%

import math

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborListOptions, System

import meshlode


# %%
# Define simple example structure having the CsCl structure and compute the reference
# values. PMEPotential by default outputs the types sorted according to the atomic
# number. Thus, we input the compound "CsCl" and "ClCs" since Cl and Cs have atomic
# numbers 17 and 55, respectively.

types = torch.tensor([17, 55])  # Cl and Cs
positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
charges = torch.tensor([-1.0, 1.0]).reshape(-1, 1)
cell = torch.eye(3)

# %%
# Define the expected values of the energy

n_atoms = len(types)
madelung = 2 * 1.7626 / math.sqrt(3)
energies_ref = -madelung * torch.ones((n_atoms, 1))

# %%
# We first define general parameters for our calculation MeshLODE.

atomic_smearing = 0.1
cell = torch.eye(3)
mesh_spacing = atomic_smearing / 4
interpolation_order = 2


# %%
# Generate neighbor list using ASE's :py:func:`neighbor_list()
# <ase.neighborlist.neighbor_list>` function.

sr_cutoff = np.sqrt(3) * 0.8
struc = Atoms(positions=positions, cell=cell, pbc=True)
nl_i, nl_j, nl_S, nl_D = neighbor_list("ijSD", struc, sr_cutoff)


# %%
# Convert ASE neighbor list into suitable format for a Metatensor system.

neighbors = TensorBlock(
    values=torch.from_numpy(nl_D.astype(np.float32).reshape(-1, 3, 1)),
    samples=Labels(
        names=[
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=torch.from_numpy(np.vstack([nl_i, nl_j, nl_S.T]).T),
    ),
    components=[Labels.range("xyz", 3)],
    properties=Labels.range("distance", 1),
)


# %%
# Define the system.

system = System(types=types, positions=positions, cell=cell)

# %%
# Attach charges to the system.

data = TensorBlock(
    values=charges,
    samples=Labels.range("atom", charges.shape[0]),
    components=[],
    properties=Labels.range("charge", charges.shape[1]),
)
system.add_data(name="charges", data=data)

# %%
# Attach ``neighbors`` to ``system`` object.

nl_options = NeighborListOptions(cutoff=sr_cutoff, full_list=True)
system.add_neighbor_list(options=nl_options, neighbors=neighbors)

pme = meshlode.metatensor.PMEPotential(
    atomic_smearing=atomic_smearing,
    mesh_spacing=mesh_spacing,
    interpolation_order=interpolation_order,
    subtract_self=True,
)
potential = pme.compute(system)

# %%
# The total energy is just the sum of all atomic energies

print(potential)

# total_energy_metatensor = torch.sum(potential[0].values)

# # %%
# # Compare against reference Madelung constant and reference energy:

# print("Using the metatensor version")
# print(f"Computed energies on each atom = {potential[0].values.tolist()}")
# print(f"Reference Madelung constant = {madelung:.3f}")
# print(f"Total energy = {total_energy_metatensor[0].values}")
