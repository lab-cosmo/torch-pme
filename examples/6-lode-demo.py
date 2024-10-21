"""
.. _example-lode-demo:

Computing LODE descriptors
==========================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This notebook demonstrates the use of some advanced features of ``torch-pme``
to compute long-distance equivariant features, cf. GRISAFI PAPER
"""

# %%

import ase
import chemiscope
import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt

import torchpme
from torchpme.lib.potentials import CoulombPotential, SplinePotential

device = "cpu"
dtype = torch.float64
rng = torch.Generator()
rng.manual_seed(42)

# import matplotlib
# matplotlib.use("widget")

# %%
# Demonstrate spline potential class

# generate reference real-space data for a pure Coulomb potential
coulomb = CoulombPotential(smearing=1.0)
x_grid = torch.logspace(-3, 2, 2000)
y_grid = coulomb.lr_from_dist(x_grid)

# create a spline potential
spline = SplinePotential(r_grid=x_grid, y_grid=y_grid, reciprocal=True)

# %%
# The real-space part of the potential matches the reference
t_grid = torch.logspace(-torch.pi, torch.pi, 100)
z_coul = coulomb.lr_from_dist(t_grid)
z_spline = spline.lr_from_dist(t_grid)

fig, ax = plt.subplots(
    1, 1, figsize=(6, 4), sharey=True, sharex=True, constrained_layout=True
)
ax.plot(t_grid, z_coul, "b-")
ax.plot(t_grid, z_spline, "r--")
ax.set_xlabel(r"$r$ / Å")
ax.set_ylabel(r"$V$ / a.u.")
ax.set_xlim(0, 100)

# %%
# Fourier-domain part (note it gets noisy)

k_grid2 = torch.logspace(-2.1, 2.1, 301)
krn_coul = coulomb.kernel_from_k_sq(k_grid2)
krn_spline = spline.kernel_from_k_sq(k_grid2)

fig, ax = plt.subplots(
    1, 1, figsize=(6, 4), sharey=True, sharex=True, constrained_layout=True
)
ax.loglog(torch.sqrt(k_grid2), krn_coul, "b-")
ax.loglog(torch.sqrt(k_grid2), krn_spline, "r--")
ax.set_xlabel(r"$k$ / Å$^{-1}$")
ax.set_ylabel(r"$V$ / a.u.")
ax.set_xlim(1e-1, 10)
ax.set_ylim(1e-8, 1e3)

# %%
# Now shows a cutoffed version
# ----------------------------
#

smearing = 1.0
cutoff = 2.0

coulomb = CoulombPotential(smearing=smearing, exclusion_radius=cutoff)
x_grid = torch.logspace(-3, 2, 2000)
y_grid = coulomb.lr_from_dist(x_grid) + coulomb.sr_from_dist(x_grid)

# create a spline potential for both the SR and LR parts
spline = SplinePotential(r_grid=x_grid, y_grid=y_grid, smearing=1.0, reciprocal=True)

# %%
# The real-space part of the potential matches the reference
t_grid = torch.logspace(-torch.pi, torch.pi, 100)
z_coul = coulomb.lr_from_dist(t_grid) + coulomb.sr_from_dist(t_grid)
z_spline = spline.lr_from_dist(t_grid)

fig, ax = plt.subplots(
    1, 1, figsize=(6, 4), sharey=True, sharex=True, constrained_layout=True
)
ax.plot(t_grid, z_coul, "b-")
ax.plot(t_grid, z_spline, "r--")
ax.set_xlabel(r"$r$ / Å")
ax.set_ylabel(r"$V$ / a.u.")
ax.set_xlim(0, 10)
ax.set_ylim(0, 0.5)

# %%
# Fourier-domain part (note it gets noisy)

k_grid2 = torch.logspace(-2.1, 4.1, 301)
krn_coul = coulomb.kernel_from_k_sq(k_grid2)
krn_spline = spline.kernel_from_k_sq(k_grid2)

fig, ax = plt.subplots(
    1, 1, figsize=(6, 4), sharey=True, sharex=True, constrained_layout=True
)
ax.semilogx(torch.sqrt(k_grid2), krn_coul, "b-")
ax.semilogx(torch.sqrt(k_grid2), krn_spline, "r--")
ax.set_xlabel(r"$k$ / Å$^{-1}$")
ax.set_ylabel(r"$V$ / a.u.")
ax.set_xlim(1e-1, 1e3)
ax.set_ylim(-10, 1e2)

# %%
# Generate a trial structure -- a distorted rocksalt structure
# with perturbed positions and charges

structure = ase.Atoms(
    positions=[
        [0, 0, 0],
        [3, 0, 0],
        [0, 3, 0],
        [3, 3, 0],
        [0, 0, 3],
        [3, 0, 3],
        [0, 3, 3],
        [3, 3, 3],
    ],
    cell=[6, 6, 6],
    symbols="NaClClNaClNaNaCl",
)

displacement = torch.normal(
    mean=0.0, std=2.5e-1, size=(len(structure), 3), generator=rng
)
structure.positions += displacement.numpy()

charges = torch.tensor(
    [[1.0], [-1.0], [-1.0], [1.0], [-1.0], [1.0], [1.0], [-1.0]],
    dtype=dtype,
    device=device,
)
charges += torch.normal(mean=0.0, std=1e-1, size=(len(charges), 1), generator=rng)
positions = torch.from_numpy(structure.positions).to(device=device, dtype=dtype)
cell = torch.from_numpy(structure.cell.array).to(device=device, dtype=dtype)

# %%
#
#

ns = torchpme.lib.kvectors.get_ns_mesh(cell, smearing * 0.5)
MI = torchpme.lib.MeshInterpolator(cell=cell, ns_mesh=ns, interpolation_nodes=3)
KF = torchpme.lib.KSpaceFilter(
    cell=cell,
    ns_mesh=ns,
    kernel=spline,
    fft_norm="backward",
    ifft_norm="forward",
)

MI.compute_weights(positions)
rho_mesh = MI.points_to_mesh(particle_weights=charges)
ivolume = torch.abs(cell.det()).pow(-1)
KF.update_mesh(cell, ns)
potential_mesh = KF.compute(rho_mesh) * ivolume

# %%

fig, ax = plt.subplots(
    1, 1, figsize=(4, 4), sharey=True, sharex=True, constrained_layout=True
)
mesh_extent = [
    0,
    cell[0, 0],
    0,
    cell[1, 1],
]

z_plot = potential_mesh[0, :, :, 0].cpu().detach().numpy()
z_plot = np.vstack([z_plot, z_plot[0, :]])  # Add first row at the bottom
z_plot = np.hstack(
    [z_plot, z_plot[:, 0].reshape(-1, 1)]
)  # Add first column at the right

z_min, z_max = (z_plot.min(), z_plot.max())

cf = ax.imshow(
    z_plot,
    extent=mesh_extent,
    vmin=z_min,
    vmax=z_max,
    origin="lower",
    interpolation="bilinear",
)

ax.set_xlabel(r"$x$ / Å")
ax.set_ylabel(r"$y$ / Å")
fig.colorbar(cf, label=r"potential / a.u.")
fig.show()

# %%
# quadrature
#


def get_theta_phi_quadrature(L):
    """
    Legendre quadrature nodes for integrals over theta, phi
    """
    quads = []
    weights = []
    for w_index in range(0, 2 * L - 1):
        w = 2 * np.pi * w_index / (2 * L - 1)
        roots_legendre_now, weights_now = scipy.special.roots_legendre(L)
        all_v = np.arccos(roots_legendre_now)
        for v, weight in zip(all_v, weights_now):
            quads.append([v, w])
            weights.append(weight)
    norm = 4 * torch.pi / np.sum(weights)
    return torch.tensor(quads), torch.tensor(weights) * norm


def get_radial_quadrature(order, R):
    """
    Generates Gauss-Legendre quadrature nodes and weights for radial integration
    in spherical coordinates over the interval [0, R].
    """
    # Step 1: Obtain Gauss-Legendre nodes and weights on [-1, 1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(order)

    # Step 2: Map nodes from [-1, 1] to [0, R]
    nodes = (R / 2) * (gl_nodes + 1)

    # Step 3: Adjust weights to account for the transformation and r^2 weighting
    weights = (R / 2) ** 3 * gl_weights * (gl_nodes + 1) ** 2

    return torch.from_numpy(nodes), torch.from_numpy(weights)


def get_full_grid(n, R):
    lm_nodes, lm_weights = get_theta_phi_quadrature(n)
    r_nodes, r_weights = get_radial_quadrature(n, R)

    full_weights = (r_weights.reshape(-1, 1) * lm_weights.reshape(1, -1)).flatten()

    cos_nodes = torch.cos(lm_nodes[:, 0]).reshape(1, -1)
    sin_nodes = torch.sin(lm_nodes[:, 0]).reshape(1, -1)
    xyz_nodes = torch.vstack(
        [
            (r_nodes.reshape(-1, 1) * cos_nodes).flatten(),
            (
                r_nodes.reshape(-1, 1) * (sin_nodes * torch.cos(lm_nodes[:, 1]))
            ).flatten(),
            (
                r_nodes.reshape(-1, 1) * (sin_nodes * torch.sin(lm_nodes[:, 1]))
            ).flatten(),
        ]
    ).T

    return xyz_nodes, full_weights


# %%
xyz, w = get_full_grid(3, 2.5)
# %%

points = positions[3] + xyz

MI.compute_weights(points)
# %%

pots = MI.mesh_to_points(potential_mesh).squeeze()

# %%

dummy = ase.Atoms(positions=points.numpy(), symbols="H" * len(points))
chemiscope.show(
    frames=[structure + dummy],
    properties={
        "potential": {
            "target": "atom",
            "values": np.concatenate([[0] * len(positions), pots.flatten().numpy()]),
        }
    },
    mode="structure",
    settings=chemiscope.quick_settings(
        structure_settings={
            "unitCell": True,
            "bonds": False,
            "environments": {"activated": False},
            "color": {
                "property": "potential",
                "min": -0.5,
                "max": 0.5,
                "transform": "linear",
                "palette": "seismic",
            },
        }
    ),
    environments=chemiscope.all_atomic_environments([structure + dummy]),
)

# %%
# stencils for integration

f0 = torch.ones(len(xyz))

fx = xyz[:, 0]
fy = xyz[:, 1]
fz = xyz[:, 2]

# %%
(w * f0 * pots).sum() / (w * f0).sum()

# %%
(w * fx * pots).sum() / (w * f0).sum()

# %%
(w * fy * pots).sum() / (w * f0).sum()

# %%
(w * fz * pots).sum() / (w * f0).sum()

# %%

# %%
(w * fz * fz).sum() / (w * f0).sum()
# %%

# %%
# now defines a LODE calculator