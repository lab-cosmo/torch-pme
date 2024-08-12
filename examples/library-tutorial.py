"""
Basic Tutorial for Library functions
====================================
This examples provides an illustration of the functioning of the underlaying library
functions of ``torchpme`` and the construction LODE descriptors (`Grisafi 2019
<https://doi.org/10.1063/1.5128375>`__, `Grisafi 2021
<https://doi.org/10.1039/D0SC04934D>`__, `Huguenin 2023
<10.1021/acs.jpclett.3c02375>`__). It builds the (simple and weighted) atom density for
a CsCl-type structure, computes a smeared version and the Coulomb potential, and
projects on a separate set of points.
"""

# %%

import ase
import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import torch
from metatensor.torch.atomistic import System

import torchpme


torch.set_default_dtype(torch.float64)


# plot a 3D mesh with a stack of 2D plots
def sliceplot(mesh, sz=12, cmap="viridis", vmin=None, vmax=None):
    mesh = mesh.detach().numpy()
    if vmin is None:
        vmin = mesh.min()
    if vmax is None:
        vmax = mesh.max()
    _, ax = plt.subplots(
        1,
        mesh.shape[-1],
        figsize=(sz, sz / mesh.shape[-1]),
        sharey=True,
        constrained_layout=True,
    )
    for i in range(mesh.shape[-1]):
        ax[i].matshow(mesh[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])


# %%
# Builds the structure
# --------------------
# Builds a CsCl structure by replicating the primitive cell using ase and convert it to
# a :py:class:`list` of :py:class:`metatensor.torch.atomistic.System`. We add a bit of
# noise to make it less boring!
#

positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]]) * 4
types = torch.tensor([55, 17])  # Cs and Cl
cell = torch.eye(3) * 4
ase_frame = ase.Atoms(positions=positions, cell=cell, numbers=types).repeat([2, 2, 2])
ase_frame.positions[:] += np.random.normal(size=ase_frame.positions.shape) * 0.1
charges = torch.tensor([1.0, -1.0] * 8)
system = System(
    types=torch.tensor(ase_frame.numbers),
    positions=torch.tensor(np.array(ase_frame.positions)),
    cell=torch.tensor(ase_frame.cell),
)

cs = chemiscope.show(
    frames=[ase_frame],
    mode="structure",
    settings={"structure": [{"unitCell": True, "axes": "xyz"}]},
)

if chemiscope.jupyter._is_running_in_notebook():
    from IPython.display import display

    display(cs)
else:
    cs.save("cscl.json.gz")


# %%
# MeshInterpolator
# ----------------
# ``MeshInterpolator`` serves as a utility class to compute a mesh
# representation of points, and/or to project a function defined on the
# mesh on a set of points. Computing the mesh representation is a two-step
# procedure. First, the weights associated with the interpolation of the
# point positions are evaluated, then they are combined with one or more
# list of atom weights to yield the mesh values.
#

interpol = torchpme.lib.mesh_interpolator.MeshInterpolator(
    system.cell, torch.tensor([16, 16, 16]), interpolation_order=3
)

interpol.compute_interpolation_weights(system.positions)


# %%
# We use two sets of weights: ones (giving the atom density irrespective
# of the types) and charges (giving a smooth representation of the point
# charges).
#

atom_weights = torch.ones((len(charges), 2))
atom_weights[:, 1] = charges
mesh = interpol.points_to_mesh(atom_weights)

# there are two densities
mesh.shape

# %%
# the first corresponds to plain density
sliceplot(mesh[0, :, :, :5])

# %%
# the second to the charge-weighted one
sliceplot(mesh[1, :, :, :5], cmap="seismic", vmax=1, vmin=-1)


# %%
# Fourier filter
# --------------
# This module computes a Fourier-domain filter, that can be used e.g. to
# smear the density and/or compute a 1/r^p potential field. This can also
# be easily extended to compute an arbitrary filter
#

fsc = torchpme.lib.fourier_convolution.FourierSpaceConvolution()

# %%
# plain atomic_smearing
rho_mesh = fsc.compute(
    mesh_values=mesh, cell=system.cell, potential_exponent=0, atomic_smearing=1
)

sliceplot(rho_mesh[0, :, :, :5])

# %%
# coulomb-like potential, no atomic_smearing
coulomb_mesh = fsc.compute(
    mesh_values=mesh, cell=system.cell, potential_exponent=1, atomic_smearing=0
)

sliceplot(coulomb_mesh[1, :, :, :5], cmap="seismic")


# %%
# Back-interpolation (on the same points)
# ---------------------------------------
# The same ``MeshInterpolator`` object can be used to compute a field on
# the same points used initially to generate the atom density
#

potentials = interpol.mesh_to_points(coulomb_mesh)

potentials


# %%
# Back-interpolation (on different points)
# ----------------------------------------
# In order to compute the field on a different set of points, it is
# sufficient to build another ``MeshInterpolator`` object and to compute
# it with the desired field. One can also use a different
# ``interpolation_order``, if wanted.
#

interpol_slice = torchpme.lib.mesh_interpolator.MeshInterpolator(
    system.cell, torch.tensor([16, 16, 16]), interpolation_order=4
)

# Compute a denser grid on a 2D slice
n_points = 50
x = torch.linspace(0, system.cell[0, 0], n_points + 1)[:n_points]
y = torch.linspace(0, system.cell[1, 1], n_points + 1)[:n_points]
xx, yy = torch.meshgrid(x, y, indexing="ij")

# Flatten xx and yy, and concatenate with a zero column for the z-coordinate
slice_points = torch.cat(
    (xx.reshape(-1, 1), yy.reshape(-1, 1), 0.5 * torch.ones(n_points**2, 1)), dim=1
)


# %%
interpol_slice.compute_interpolation_weights(slice_points)

coulomb_slice = interpol_slice.mesh_to_points(coulomb_mesh)

plt.contourf(
    xx,
    yy,
    coulomb_slice[:, 1].reshape(n_points, n_points).T,
    cmap="seismic",
    vmin=-1,
    vmax=1,
)
