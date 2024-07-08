"""
Compute Madelung Constants
==========================
In this tutorial we show how to calculate the Madelung constants and total electrostatic
energy of atomic structures using the :py:class:`meshlode.PMEPotential` and
:py:class:`meshlode.metatensor.PMEPotential` calculator.
"""

# %%
import math

import torch
from metatensor.torch.atomistic import System

import meshlode


# %%
# Define simple example structure having the CsCl structure and compute the reference
# values. PMEPotential by default outputs the types sorted according to the atomic
# number. Thus, we input the compound "CsCl" and "ClCs" since Cl and Cs have atomic
# numbers 17 and 55, respectively.
types = torch.tensor([17, 55])  # Cl and Cs
positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
charges = torch.tensor([-1.0, 1.0])
cell = torch.eye(3)

# %%
# Define the expected values of the energy
n_atoms = len(types)
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
# Compute features using

MP = meshlode.PMEPotential(
    atomic_smearing=atomic_smearing,
    mesh_spacing=mesh_spacing,
    interpolation_order=interpolation_order,
    subtract_self=True,
)
potentials_torch = MP.compute(types=types, positions=positions, cell=cell)

# %%
# The "potentials" that have been computed so far are not the actual electrostatic
# potentials. For instance, for the Cs atom, we are separately storing the contributions
# to the potential (at the location of the Cs atom) from the Cs atoms and Cl atoms
# separately. Thus, to get the Madelung constant, we need to take a linear combination
# of these "potentials" weighted by the charges of the atoms.

atomic_energies_torch = torch.zeros((n_atoms, 1))
for idx_c in range(n_atoms):
    for idx_n in range(n_atoms):
        # The coulomb potential between atoms i and j is charge_i * charge_j / d_ij
        # The features are simply computing a pure 1/r potential with no prefactors.
        # Thus, to compute the energy between atoms of types i and j, we need to
        # multiply by the charges of i and j.
        print(charges[idx_c] * charges[idx_n], potentials_torch[idx_n, idx_c])
        atomic_energies_torch[idx_c] += (
            charges[idx_c] * charges[idx_n] * potentials_torch[idx_c, idx_n]
        )

# %%
# The total energy is just the sum of all atomic energies
total_energy_torch = torch.sum(atomic_energies_torch)

# %%
# Compare against reference Madelung constant and reference energy:
print("Using the torch version")
print(f"Computed energies on each atom = {atomic_energies_torch.tolist()}")
print(f"Reference Madelung constant = {madelung:.3f}")
print(f"Total energy = {total_energy_torch:.3f}\n")


# %%
# Computation using ``meshlode.metatensor``
# -----------------------------------------
# We now compute the same constants using the metatensor based calculator. To achieve
# this we first store our system parameters like the ``types``, ``positions`` and the
# ``cell`` defined above into a :py:class:`metatensor.torch.atomistic.System` class.

system = System(types=types, positions=positions, cell=cell)

MP = meshlode.metatensor.PMEPotential(
    atomic_smearing=atomic_smearing,
    mesh_spacing=mesh_spacing,
    interpolation_order=interpolation_order,
    subtract_self=True,
)
potential_metatensor = MP.compute(system)


# %%
# To get the Madelung constant, we again need to take a linear combination
# of the "potentials" weighted by the charges of the atoms.

atomic_energies_metatensor = torch.zeros((n_atoms, 1))
for idx_c, c in enumerate(types):
    for idx_n, n in enumerate(types):
        # Take the coefficients with the correct center atom and neighbor atom types
        block = potential_metatensor.block(
            {"center_type": int(c), "neighbor_type": int(n)}
        )

        # The coulomb potential between atoms i and j is charge_i * charge_j / d_ij
        # The features are simply computing a pure 1/r potential with no prefactors.
        # Thus, to compute the energy between atoms of types i and j, we need to
        # multiply by the charges of i and j.
        print(c, n, charges[idx_c] * charges[idx_n], block.values[0, 0])
        atomic_energies_metatensor[idx_c] += (
            charges[idx_c] * charges[idx_n] * block.values[0, 0]
        )

# %%
# The total energy is just the sum of all atomic energies
total_energy_metatensor = torch.sum(atomic_energies_metatensor)

# %%
# Compare against reference Madelung constant and reference energy:
print("Using the metatensor version")
print(f"Computed energies on each atom = {atomic_energies_metatensor.tolist()}")
print(f"Reference Madelung constant = {madelung:.3f}")
print(f"Total energy = {total_energy_metatensor:.3f}")
