"""
.. _example-kspace-demo:

Examples of the ``KSpaceFilter`` class
==========================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This notebook demonstrates the use of the 
:py:class:`KSpaceFilter <torchpme.lib.KSpaceFilter>` class
to transform a density by applying a scalar filter in reciprocal space.
"""

# %%
# Import dependencies

import ase
import chemiscope
import numpy as np
import torch
from matplotlib import pyplot as plt

import torchpme


device = "cpu"
dtype = torch.float64
torch.manual_seed(12345)

# %%
# Demonstrates the application of a k-space filter
# ------------------------------------------------
#
# Defines a fairly rugged function, and applies a
# smoothening filter. We start creating a grid
# (we use a :py:class:`MeshInterpolator` object for
# simplicity to generate the grid) and computing a
# sharp Gaussian field in the :math:`xy` plane.

cell = torch.eye(3) * 6.0
ns_mesh = torch.tensor([9, 9, 9])
MI = torchpme.lib.MeshInterpolator(cell, ns_mesh, interpolation_order=2)
xyz_mesh = MI.get_mesh_xyz()

mesh_value = (
    np.exp(-4 * ((cell[0, 0] / 2 - xyz_mesh) ** 2)[..., :2].sum(axis=-1))
).reshape(1, *xyz_mesh.shape[:-1])

# %%
# We define and apply a Gaussian smearing filter

KF = torchpme.lib.KSpaceFilter(cell, ns_mesh)


# this is the filter function. NB it is applied
# to the *squared k vector norm*
def filter(k2):
    sigma2 = 1
    return torch.exp(-k2 * sigma2 / 2)


KF.set_filter_mesh(filter)

mesh_filtered = KF.compute(mesh_value)

# %%
# Visualize the effect of the filter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The Gaussian smearing enacted by the filter can be
# easily visualized taking a slice of the mesh (we make
# it periodic for a cleaner view)

fig, ax = plt.subplots(
    1, 2, figsize=(8, 4), sharey=True, sharex=True, constrained_layout=True
)
mesh_extent = [
    0,
    cell[0, 0],
    0,
    cell[1, 1],
]

z_plot = mesh_value[0, :, :, 0].detach().numpy()
z_plot = np.vstack([z_plot, z_plot[0, :]])  # Add first row at the bottom
z_plot = np.hstack(
    [z_plot, z_plot[:, 0].reshape(-1, 1)]
)  # Add first column at the right

z_min, z_max = (z_plot.min(), z_plot.max())

cf = ax[0].imshow(
    z_plot,
    extent=mesh_extent,
    vmin=z_min,
    vmax=z_max,
    origin="lower",
    interpolation="bilinear",
)

z_plot = mesh_filtered[0, :, :, 0].detach().numpy()
z_plot = np.vstack([z_plot, z_plot[0, :]])  # Add first row at the bottom
z_plot = np.hstack(
    [z_plot, z_plot[:, 0].reshape(-1, 1)]
)  # Add first column at the right

cf_fine = ax[1].imshow(
    z_plot,
    extent=mesh_extent,
    vmin=z_min,
    vmax=z_max,
    origin="lower",
    interpolation="bilinear",
)
ax[0].set_xlabel("x / Å")
ax[1].set_xlabel("x / Å")
ax[0].set_ylabel("y / Å")
ax[0].set_title(r"input values")
ax[1].set_title(r"filtered")
fig.colorbar(cf_fine, label=r"density / e/Å$^3$")
fig.show()

# %%
# We can also show the filter in action in 3D: create
# a grid of dummy atoms corresponding to the mesh
# points, and colored according to the function
# value. Use the `chemiscope` option panel to
# switch between the sharp input and the filtered
# values.

dummy = ase.Atoms(
    positions=xyz_mesh.reshape(-1, 3),
    symbols="H" * len(xyz_mesh.reshape(-1, 3)),
    cell=cell,
)
chemiscope.show(
    frames=[dummy],
    properties={
        "input value": {
            "target": "atom",
            "values": mesh_value.detach().numpy().flatten(),
        },
        "filtered value": {
            "target": "atom",
            "values": mesh_filtered.detach().numpy().flatten(),
        },
    },
    mode="structure",
    settings=chemiscope.quick_settings(
        structure_settings={
            "unitCell": True,
            "bonds": False,
            "environments": {"activated": False},
            "color": {
                "property": "input value",
                "transform": "linear",
                "palette": "viridis",
            },
        }
    ),
    environments=chemiscope.all_atomic_environments([dummy]),
)
