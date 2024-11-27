"""
Parameter tuning for range-separated models
===========================================

.. currentmodule:: torchpme

We explain and demonstrate parameter tuning for Ewald and PME
"""

# %%

from time import time

import ase
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import vesin.torch as vesin

import torchpme
from torchpme.lib.kvectors import get_ns_mesh

DTYPE = torch.float64

get_ipython().run_line_magic("matplotlib", "inline")

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
charges = torch.tensor([+1.0, -1, -1, -1, +1, +1, +1, -1], dtype=DTYPE).reshape(-1, 1)
cell = 2 * torch.eye(3, dtype=DTYPE)
madelung_ref = 1.7475645946
num_formula_units = 4

atoms = ase.Atoms("NaCl3Na3Cl", positions, cell=cell)


# %%
# compute and compare with reference

smearing = 0.5
pme_params = {"mesh_spacing": 0.5, "interpolation_nodes": 4}
pme_params_in_ns_mesh = {
    "ns_mesh": torch.tensor(get_ns_mesh(cell, pme_params["mesh_spacing"])),
    "interpolation_nodes": torch.tensor(4),
}
cutoff = 5.0

max_cutoff = 32.0

nl = vesin.NeighborList(cutoff=max_cutoff, full_list=False)
i, j, S, d = nl.compute(points=positions, box=cell, periodic=True, quantities="ijSd")
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
madelung = (-energy / num_formula_units).flatten().item()

# this is the estimated error
error_bounds = torchpme.utils.tuning.pme.PMEErrorBounds(
    (charges**2).sum(), cell, positions
)

estimated_error = error_bounds(
    torch.tensor(max_cutoff), torch.tensor(smearing), **pme_params_in_ns_mesh
).item()
print(
    f"""
Computed madelung constant: {madelung}
Actual error: {madelung-madelung_ref}
Estimated error: {estimated_error}
"""
)

# %%
# now set up a testing framework


def timed_madelung(cutoff, smearing, mesh_spacing, interpolation_nodes):
    assert cutoff <= max_cutoff

    filter_idx = torch.where(neighbor_distances <= cutoff)
    filter_indices = neighbor_indices[filter_idx]
    filter_distances = neighbor_distances[filter_idx]

    pme = torchpme.PMECalculator(
        potential=torchpme.CoulombPotential(smearing=smearing),
        mesh_spacing=mesh_spacing,
        interpolation_nodes=interpolation_nodes,
    )
    start = time()
    potential = pme(
        charges=charges,
        cell=cell,
        positions=positions,
        neighbor_indices=filter_indices,
        neighbor_distances=filter_distances,
    )
    energy = charges.T @ potential
    madelung = (-energy / num_formula_units).flatten().item()
    end = time()

    return madelung, end - start


smearing_grid = torch.logspace(-1, 0.5, 8)
spacing_grid = torch.logspace(-1, 0.5, 9)
results = np.zeros((len(smearing_grid), len(spacing_grid)))
timings = np.zeros((len(smearing_grid), len(spacing_grid)))
bounds = np.zeros((len(smearing_grid), len(spacing_grid)))
for ism, smearing in enumerate(smearing_grid):
    for isp, spacing in enumerate(spacing_grid):
        results[ism, isp], timings[ism, isp] = timed_madelung(8.0, smearing, spacing, 4)
        bounds[ism, isp] = error_bounds(
            torch.tensor(8.0),
            torch.tensor(smearing),
            get_ns_mesh(cell, spacing),
            torch.tensor(4),
        )

# %%
# plot

vmin = 1e-12
vmax = 2
levels = np.geomspace(vmin, vmax, 30)

fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True, constrained_layout=True)
contour = ax[0].contourf(
    spacing_grid,
    smearing_grid,
    bounds,
    vmin=vmin,
    vmax=vmax,
    levels=levels,
    norm=mpl.colors.LogNorm(),
    extend="both",
)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$\sigma$ / Å")
ax[0].set_xlabel(r"spacing / Å")
ax[0].set_title("estimated error")
cbar = fig.colorbar(contour, ax=ax[1], label="error")
cbar.ax.set_yscale("log")

contour = ax[1].contourf(
    spacing_grid,
    smearing_grid,
    np.abs(results - madelung_ref),
    vmin=vmin,
    vmax=vmax,
    levels=levels,
    norm=mpl.colors.LogNorm(),
    extend="both",
)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"spacing / Å")
ax[1].set_title("actual error")

contour = ax[2].contourf(
    spacing_grid,
    smearing_grid,
    timings,
    levels=np.geomspace(1e-3, 2e-2, 20),
    norm=mpl.colors.LogNorm(),
)
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].set_ylabel(r"$\sigma$ / Å")
ax[2].set_xlabel(r"spacing / Å")
ax[2].set_title("actual timing")
cbar = fig.colorbar(contour, ax=ax[2], label="time / s")
cbar.ax.set_yscale("log")

# cbar.ax.set_yscale('log')


# %%
#
# a good heuristic is to keep cutoff/sigma constant (easy to
# determine error limit) to see how timings change

smearing_grid = torch.logspace(-1, 0.5, 8)
spacing_grid = torch.logspace(-1, 0.5, 9)
results = np.zeros((len(smearing_grid), len(spacing_grid)))
timings = np.zeros((len(smearing_grid), len(spacing_grid)))
for ism, smearing in enumerate(smearing_grid):
    for isp, spacing in enumerate(spacing_grid):
        madelung, timing = timed_madelung(smearing * 8, smearing, spacing, 4)
        results[ism, isp] = madelung
        timings[ism, isp] = timing


# %%
# plot

fig, ax = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
contour = ax[0].contourf(
    spacing_grid, smearing_grid, np.log10(np.abs(results - madelung_ref))
)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$\sigma$ / Å")
ax[0].set_xlabel(r"spacing / Å")
cbar = fig.colorbar(contour, ax=ax[0], label="log10(error)")

contour = ax[1].contourf(spacing_grid, smearing_grid, np.log10(timings))
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"spacing / Å")
cbar = fig.colorbar(contour, ax=ax[1], label="log10(time / s)")

# %%

EB = torchpme.utils.tuning.pme.PMEErrorBounds((charges**2).sum(), cell, positions)

# %%
v, t = timed_madelung(cutoff=5, smearing=1, mesh_spacing=1, interpolation_nodes=4)
print(
    v - madelung_ref,
    t,
    EB.forward(
        cutoff=torch.tensor(5),
        smearing=torch.tensor(1),
        ns_mesh=get_ns_mesh(cell, 1),
        interpolation_nodes=torch.tensor(4),
    ).item(),
)

# %%


from scipy.optimize import minimize


def loss(x, target_accuracy):
    cutoff, smearing, mesh_spacing = x
    value, duration = timed_madelung(
        cutoff=cutoff,
        smearing=smearing,
        mesh_spacing=mesh_spacing,
        interpolation_nodes=4,
    )
    estimated_error = error_bounds(
        cutoff=torch.tensor(cutoff),
        smearing=torch.tensor(smearing),
        ns_mesh=get_ns_mesh(cell, mesh_spacing),
        interpolation_nodes=torch.tensor(4),
    )
    tgt_loss = max(
        0, np.log(estimated_error / madelung_ref / target_accuracy)
    )  # relu on the accuracy
    print(x, estimated_error.item(), np.abs(madelung - value), duration)
    return tgt_loss * 10 + duration


initial_guess = [9, 0.3, 5]
result = minimize(
    loss,
    initial_guess,
    args=(1e-8),
    method="Nelder-Mead",
    options={"disp": True, "maxiter": 200},
)


# %%

result
# %%
timed_madelung(cutoff=2.905, smearing=0.7578, mesh_spacing=5.524, interpolation_nodes=4)
# %%

madelung_ref
# %%
error_bounds(9, 0.5, 1, 4)
# %%
