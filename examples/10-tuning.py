r"""
Parameter tuning for range-separated models
===========================================

.. currentmodule:: torchpme

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

Metods to compute efficiently a long-range potential :math:`v(r)`
usually rely on partitioning it into a short-range part, evaluated
as a sum over neighbor pairs, and a long-range part evaluated
in reciprocal space

.. math::

    v(r)= v_{\mathrm{SR}}(r) + v_{\mathrm{LR}}(r)

The overall cost depend on the balance of multiple factors, that
we summarize here briefly to explain how the cost of evaluating
:math:`v(r)` can be minimized, either manually or automatically.
"""

# %%
# Import modules

from time import time

import ase
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import vesin.torch as vesin

import torchpme
from torchpme.tuning.pme import PMEErrorBounds
from torchpme.tuning.tuner import TuningTimings

device = "cpu"
dtype = torch.float64
rng = torch.Generator()
rng.manual_seed(42)

# get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore # noqa

# %%
# Set up a test system, a supercell containing atoms with a NaCl structure

madelung_ref = 1.7475645946
structure = ase.Atoms(
    positions=[
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    cell=[2, 2, 2],
    symbols="NaClClNaClNaNaCl",
)
structure = structure.repeat([3, 3, 3])
num_formula_units = len(structure) // 2

# Uncomment these to add a displacement (energy won't match the Madelung constant)
# displacement = torch.normal(
#    mean=0.0, std=2.5e-1, size=(len(structure), 3), generator=rng
# )
# structure.positions += displacement.numpy()

positions = torch.from_numpy(structure.positions).to(device=device, dtype=dtype)
cell = torch.from_numpy(structure.cell.array).to(device=device, dtype=dtype)

charges = torch.tensor(
    [[1.0], [-1.0], [-1.0], [1.0], [-1.0], [1.0], [1.0], [-1.0]]
    * (len(structure) // 8),
    dtype=dtype,
    device=device,
).reshape(-1, 1)

# Uncomment these to randomize charges (energy won't match the Madelung constant)
# charges += torch.normal(mean=0.0, std=1e-1, size=(len(charges), 1), generator=rng)


# %%
# Demonstrate errors and timings for PME
# --------------------------------------
#
# To set up a PME calculation, we need to define its basic parameters and
# setup a few preliminary quantities.
#
# First, we need to evaluate the neighbor list; this is usually pre-computed
# by the code that calls `torch-pme`, and entails the first key parameter:
# the cutoff used to compute the real-space potential :math:`v_\mathrm{SR}(r)`


max_cutoff = 16.0

# use `vesin`
nl = vesin.NeighborList(cutoff=max_cutoff, full_list=False)
i, j, S, d = nl.compute(points=positions, box=cell, periodic=True, quantities="ijSd")
neighbor_indices = torch.stack([i, j], dim=1)
neighbor_shifts = S
neighbor_distances = d

# %%
#
# The PME calculator has a few further parameters: ``smearing``, that determines
# aggressive is the smoothing of the point charges. This makes the reciprocal-space
# part easier to compute, but makes :math:`v_\mathrm{SR}(r)` decay more slowly,
# and error that we shall investigate further later on.
# The mesh parameters involve both the spacing and the order of the interpolation
# used. Note that here we use :class:`CoulombPotential`, that computes a simple
# :math:`1/r` electrostatic interaction.

smearing = 1.0
pme_params = {"mesh_spacing": 1.0, "interpolation_nodes": 4}

pme = torchpme.PMECalculator(
    potential=torchpme.CoulombPotential(smearing=smearing),
    **pme_params,  # type: ignore[arg-type]
)

# %%
# Run the calculator
# ~~~~~~~~~~~~~~~~~~
#
# We combine the structure data and the neighbor list information to
# compute the potential at the particle positions, and then the
# energy

potential = pme(
    charges=charges,
    cell=cell,
    positions=positions,
    neighbor_indices=neighbor_indices,
    neighbor_distances=neighbor_distances,
)

energy = charges.T @ potential
madelung = (-energy / num_formula_units).flatten().item()

# %%
# Compute error bounds (and timings)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here we calculate the potential energy of the system, and compare it with the
# madelung constant to calculate the error. This is the actual error. Then we use
# the :class:`torchpme.tuning.pme.PMEErrorBounds` to calculate the error bound for
# PME. 
# Error bounds are computed explicitly for a target structure
error_bounds = PMEErrorBounds(charges, cell, positions)

estimated_error = error_bounds(cutoff=max_cutoff, smearing=smearing, **pme_params).item()

# %%
# ... and a similar class can be used to estimate the timings, that are assessed
# based on a calculator (that should be initialized with the same parameters)
timings = TuningTimings(
    charges,
    cell,
    positions,
    neighbor_indices=neighbor_indices,
    neighbor_distances=neighbor_distances,
    run_backward=True,
)
estimated_timing = timings(pme)

# %%
# The error bound is estimated for the force acting on atoms, and is 
# expressed in force units - hence, the comparison with the Madelung constant 
# error can only be qualitative.

print(
    f"""
Computed madelung constant: {madelung}
Actual error: {madelung - madelung_ref}
Estimated error: {estimated_error}
Timing: {estimated_timing} seconds
"""
)

# %%
# Optimizing the parameters of PME
# --------------------------------
#
# There are many parameters that enter the implementation
# of a range-separated calculator like PME, and it is necessary
# to optimize them to obtain the best possible accuracy/cost tradeoff.
# In most practical use cases, the cutoff is dictated by the external
# calculator and is treated as a fixed parameter. In cases where 
# performance is critical, one may want to optimize this separately,
# which can be achieved easily with a grid or binary search.
# 
# We can set up easily a brute-force evaluation of the error as a 
# function of these parameters, and use it to guide the design of 
# a more sophisticated optimization protocol. 

def filter_neighbors(cutoff, neighbor_indices, neighbor_distances):
    
    assert cutoff <= max_cutoff

    filter_idx = torch.where(neighbor_distances <= cutoff)

    return neighbor_indices[filter_idx], neighbor_distances[filter_idx]

def timed_madelung(cutoff, smearing, mesh_spacing, interpolation_nodes):
    filter_indices, filter_distances = filter_neighbors(cutoff, neighbor_indices, neighbor_distances)

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
        bounds[ism, isp] = error_bounds(8.0, smearing, spacing, 4)

# %%
# We now plot the error landscape. The estimated error can be seen as a upper bound of
# the actual error. Though the magnitude of the estimated error is higher than the
# actual error, the trend is the same. Also, from the timing results, we can see that
# the timing increases as the spacing decreases, while the smearing does not affect the
# timing, because the interactions are computed up to the fixed cutoff regardless of
# whether :math:`v_\mathrm{sr}(r)` is negligible or large.

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
    levels=np.geomspace(1e-2, 5e-1, 20),
    norm=mpl.colors.LogNorm(),
)
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].set_ylabel(r"$\sigma$ / Å")
ax[2].set_xlabel(r"spacing / Å")
ax[2].set_title("actual timing")
cbar = fig.colorbar(contour, ax=ax[2], label="time / s")
cbar.ax.set_yscale("log")

# %%

torch.Tensor([5.0]*len(smearing_grid))
smearing_grid 

# %%
# Optimizing the smearing
# ~~~~~~~~~~~~~~~~~~~~~~~
# The error is a sum of an error on the real-space evaluation of the 
# short-range potential, and of a long-range error. Considering the 
# cutoff as given, the short-range error is determined easily by how
# quickly :math:`v_\mathrm{sr}(r)` decays to zero, which depends on 
# the Gaussian smearing.

smearing_grid = torch.logspace(-0.6, 1, 20)
err_vsr_grid = error_bounds.err_rspace(smearing_grid, torch.tensor([5.0]))
err_vlr_grid_4 = [ error_bounds.err_kspace(torch.tensor([s]), 
                                         torch.tensor([1.0]), 
                                         torch.tensor([4], dtype=int)) for s in smearing_grid ] 
err_vlr_grid_2 = [ error_bounds.err_kspace(torch.tensor([s]), 
                                         torch.tensor([1.0]), 
                                         torch.tensor([3], dtype=int)) for s in smearing_grid ] 

fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.loglog(smearing_grid, err_vsr_grid, 'r-', label="real-space")
ax.loglog(smearing_grid, err_vlr_grid_4, 'b-', label="k-space (spacing: 1Å, n.int.: 4)")
ax.loglog(smearing_grid, err_vlr_grid_2, 'c-', label="k-space (spacing: 1Å, n.int.: 2)")
ax.set_ylabel(r"estimated error / a.u.")
ax.set_xlabel(r"smearing / Å")
ax.set_title("cutoff = 5.0 Å")
ax.set_ylim(1e-20,2)
ax.legend()

# %%
# Given the simple, monotonic and fast-varying trend for the real-space error, 
# it is easy to pick the optimal smearing as the value corresponding to roughly 
# half of the target error -e.g. for a target accuracy of :math:`1e^{-5}`, 
# one would pick a smearing of about 1Å. 

# %%
# Optimizing mesh and interpolation order
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %%
# word bank

smearing_grid = torch.logspace(-1, 0.5, 8)
spacing_grid = torch.logspace(-1, 0.5, 9)
results = np.zeros((len(smearing_grid), len(spacing_grid)))
timings = np.zeros((len(smearing_grid), len(spacing_grid)))
bounds = np.zeros((len(smearing_grid), len(spacing_grid)))
for ism, smearing in enumerate(smearing_grid):
    for isp, spacing in enumerate(spacing_grid):
        results[ism, isp], timings[ism, isp] = timed_madelung(8.0, smearing, spacing, 4)
        bounds[ism, isp] = error_bounds(8.0, smearing, spacing, 4)

# %%
# We now plot the error landscape. The estimated error can be seen as a upper bound of
# the actual error. Though the magnitude of the estimated error is higher than the
# actual error, the trend is the same. Also, from the timing results, we can see that
# the timing increases as the spacing decreases, while the smearing does not affect the
# timing.

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
# A good heuristic is to keep cutoff/sigma constant (easy to determine error limit,
# also the dominating term in the real space error) to see how timings change.

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
# Now we again plot the error landscape and the timing. The error is now dominated by
# The ratio of the smearing and the spacing. The larger ratio, the smaller error. The
# timing is the opposite. Thus, the error and timing are anti-correlated, to a certain
# extent. In order to achieve a balance between the speed and accuracy, we offer a
# auto-tuning feature. See :class:`torchpme.tuning.tuner.GridSearchTuner`.

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

EB = torchpme.tuning.pme.PMEErrorBounds((charges**2).sum(), cell, positions)

# %%
v, t = timed_madelung(cutoff=5, smearing=1, mesh_spacing=1, interpolation_nodes=4)
print(
    v - madelung_ref,
    t,
    EB.forward(cutoff=5, smearing=1, mesh_spacing=1, interpolation_nodes=4).item(),
)

# %%
