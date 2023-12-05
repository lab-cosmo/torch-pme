"""
Compute Madelung Constants
==========================

In this tutorial we show how to calculate the Madelung constants and total electrostatic
energy of atomic structures using the :py:class:`meshlode.MeshPotential` and
:py:class:`meshlode.metatensor.MeshPotential` calculator.
"""

# %%
import math

import torch

import meshlode


# %%
# Define simple example structure having the CsCl structure and compute the reference
# values

positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
atomic_numbers = torch.tensor([55, 17])  # Cs and Cl
cell = torch.eye(3)
charges = torch.tensor([1.0, -1.0])
frame = meshlode.System(
    species=atomic_numbers, positions=positions, cell=torch.eye(3)
)

n_atoms = len(positions)
madelung = 2 * 1.7626 / math.sqrt(3)
energies_ref = -madelung * torch.ones((n_atoms, 1))

# %%
# We first define general parameters for our calculation MeshLODE

atomic_smearing = 0.1
cell = torch.eye(3)
mesh_spacing = atomic_smearing / 4
interpolation_order = 2

# %%
# Computation using ``meshlode``
# ------------------------------
#
# Compute features using

MP = meshlode.MeshPotential(
    atomic_smearing=atomic_smearing,
    mesh_spacing=mesh_spacing,
    interpolation_order=interpolation_order,
    subtract_self=True,
)
potentials_mesh = MP.compute(frame)

# %%
# The "potentials" that have been computed so far are not the actual electrostatic
# potentials. For instance, for the Cs atom, we are separately storing the contributions
# to the potential (at the location of the Cs atom) from the Cs atoms and Cl atoms
# separately. Thus, to get the Madelung constant, we need to take a linear combination
# of these "potentials" weighted by the charges of the atoms.

atomic_energies = torch.zeros((n_atoms, 1))
for idx_c in range(n_atoms):
    for idx_n in range(n_atoms):
        # The coulomb potential between atoms i and j is charge_i * charge_j / d_ij
        # The features are simply computing a pure 1/r potential with no prefactors.
        # Thus, to compute the energy between atoms of species i and j, we need to
        # multiply by the charges of i and j.
        atomic_energies[idx_c] += (
            charges[idx_c] * charges[idx_n] * potentials_mesh[idx_c, idx_n]
        )

# %%
# The total energy is just the sum of all atomic energies
total_energy = torch.sum(atomic_energies)

# %%
# Compare against reference Madelung constant and reference energy:
print("Using the torch version")
print(f"Computed energies on each atom = {atomic_energies.tolist()}")
print(f"Reference Madelung constant = {madelung:.3f}")
print(f"Total energy = {total_energy:.3f}\n")


# %%
# Computation using ``meshlode.metatensor``
# -----------------------------------------
#
# We now compute the same constants using the metatensor based calculator

MP = meshlode.metatensor.MeshPotential(
    atomic_smearing=atomic_smearing,
    mesh_spacing=mesh_spacing,
    interpolation_order=interpolation_order,
    subtract_self=True,
)
potentials_mesh = MP.compute(frame)


# %%
# To get the Madelung constant, we again need to take a linear combination
# of the "potentials" weighted by the charges of the atoms.

atomic_energies = torch.zeros((n_atoms, 1))
for idx_c, c in enumerate(atomic_numbers):
    for idx_n, n in enumerate(atomic_numbers):
        # Take the coefficients with the correct center atom and neighbor atom species
        block = potentials_mesh.block(
            {"species_center": int(c), "species_neighbor": int(n)}
        )

        # The coulomb potential between atoms i and j is charge_i * charge_j / d_ij
        # The features are simply computing a pure 1/r potential with no prefactors.
        # Thus, to compute the energy between atoms of species i and j, we need to
        # multiply by the charges of i and j.
        atomic_energies[idx_c] += charges[idx_c] * charges[idx_n] * block.values[0, 0]

# %%
# The total energy is just the sum of all atomic energies
total_energy = torch.sum(atomic_energies)

# %%
# Compare against reference Madelung constant and reference energy:
print("Using the metatensor version")
print(f"Computed energies on each atom = {atomic_energies.tolist()}")
print(f"Reference Madelung constant = {madelung:.3f}")
print(f"Total energy = {total_energy:.3f}")
