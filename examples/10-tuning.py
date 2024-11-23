"""
Parameter tuning for range-separated models
===========================================

.. currentmodule:: torchpme

We explain and demonstrate parameter tuning for Ewald and PME
"""

# %%
from typing import Optional

import ase
import chemiscope
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import torch
import vesin.torch as vesin
from time import time
import torchpme

DTYPE=torch.float64

get_ipython().run_line_magic('matplotlib', 'inline')

# %%

positions = torch.tensor(
    [
        [0.0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=DTYPE,
)
charges = torch.tensor([+1.0, -1, -1, -1, +1, +1, +1, -1], dtype=DTYPE).reshape(-1,1)
cell = 2 * torch.eye(3, dtype=DTYPE)
madelung_ref = 1.7475645946
num_formula_units = 4

atoms = ase.Atoms("NaCl3Na3Cl", positions, cell=cell)



# %%
# compute and compare with reference

smearing = 0.5
pme_params = { "mesh_spacing": 0.5, "interpolation_nodes": 4}
cutoff = 5.0

max_cutoff=20.0

nl = vesin.NeighborList(cutoff=max_cutoff, full_list=False)
i, j, S, d = nl.compute(
    points=positions, box=cell, periodic=True, quantities="ijSd"
)
neighbor_indices = torch.stack([i, j], dim=1)
neighbor_shifts = S
neighbor_distances = d


pme = torchpme.PMECalculator(
    potential=torchpme.CoulombPotential(smearing=smearing), **pme_params
)
potential = pme(
    charges=charges,
    cell=cell,
    positions=positions,
    neighbor_indices=neighbor_indices,
    neighbor_distances=neighbor_distances,
)

energy = charges.T @ potential
madelung = (-energy/num_formula_units).flatten().item()

print(madelung, madelung_ref)

# %%
# now set up a testing framework

def timed_madelung(cutoff, smearing, mesh_spacing, interpolation_nodes):
    assert cutoff <= max_cutoff

    filter_idx = torch.where(neighbor_distances<=cutoff)
    filter_indices = neighbor_indices[filter_idx]
    filter_distances = neighbor_distances[filter_idx]

    start = time()
    pme = torchpme.PMECalculator(
        potential=torchpme.CoulombPotential(smearing=smearing), 
        mesh_spacing=mesh_spacing,
        interpolation_nodes=interpolation_nodes
    )
    potential = pme(
        charges=charges,
        cell=cell,
        positions=positions,
        neighbor_indices=filter_indices,
        neighbor_distances=filter_distances,
    )
    energy = charges.T @ potential
    madelung = (-energy/num_formula_units).flatten().item()
    end = time()

    return madelung, end-start

print(timed_madelung(5.0, 1.0, 0.1, 4))

smearing_grid = torch.logspace(-1, 0.5, 8)
spacing_grid = torch.logspace(-1, 0.5, 9)
results = np.zeros((len(smearing_grid), len(spacing_grid)))
timings = np.zeros((len(smearing_grid), len(spacing_grid)))
for ism, smearing in enumerate(smearing_grid):
    for isp, spacing in enumerate(spacing_grid):
        madelung, timing = timed_madelung(8.0, smearing, spacing, 4)
        results[ism, isp] = madelung
        timings[ism, isp] = timing

# %%
# plot

fig, ax = plt.subplots(1,2,figsize=(7,3), constrained_layout=True)
contour = ax[0].contourf(spacing_grid, smearing_grid, np.log10(np.abs(results-madelung_ref)))
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylabel(r"$\sigma$ / Å")
ax[0].set_xlabel(r"spacing / Å")
cbar = fig.colorbar(contour, ax=ax[0], label="log10(error)")

contour = ax[1].contourf(spacing_grid, smearing_grid, np.log10(timings))
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_ylabel(r"$\sigma$ / Å")
ax[1].set_xlabel(r"spacing / Å")
cbar = fig.colorbar(contour, ax=ax[1], label="log10(time / s)")

#cbar.ax.set_yscale('log')


# %%
#
# a good heuristic is to keep cutoff/sigma constant (easy to 
# determine error limit) to see how timings change


