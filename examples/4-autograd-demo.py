"""
.. _example-kspace-demo:

Custom ``torch-pme`` models with automatic differentiation
==========================================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

TBD - esplanation

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
rng = np.random.default_rng(12345)

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

# %%
#
# We now slightly displace the atoms from their initial positions randomly based on a
# Gaussian distribution.


displacement = rng.normal(loc=0.0, scale=2.5e-1, size=(len(structure), 3))
structure.positions += displacement

charges_np = np.array(
    [
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ]
)
charges_np += rng.normal(scale=0.1, size=(len(charges_np), 1))


# %%
#
# We also define the charges, with a bit of noise for good measure. (NB: the structure
# won't be charge neutral but it does not matter for this example)


# %%
#
# We now use :py:class:`MeshInterpolator <torchpme.lib.MeshInterpolator>` to project
# atomic positions on a grid. Note that ideally this represents a sharp density peaked
# at atomic positions, so the degree of smoothening depends on the grid resolution (as
# well as on the interpolation order)
#
# We demonstrate this by computing a projection on two grids with 3 and 7 mesh points.


positions = torch.from_numpy(structure.positions).to(device=device, dtype=dtype)
charges = torch.from_numpy(charges_np).to(device=device, dtype=dtype)
cell = torch.from_numpy(structure.cell.array).to(device=device, dtype=dtype)

charges.requires_grad_(True)
ns = torch.tensor([5, 5, 5])
MI = torchpme.lib.MeshInterpolator(
    cell=cell,
    ns_mesh=ns,
    interpolation_order=3,
)
MI.compute_interpolation_weights(positions)
mesh = MI.points_to_mesh(charges)


# %%
# Adjustable and multi-channel filters
# ------------------------------------
# :py:class:`KSpaceFilter <torchpme.lib.KSpaceFilter>` can
# also be applied for more complicated use cases. For
# instance, one can apply multiple filters to multiple
# real-space mesh channels, and use a
# :py:class:`torch.nn.Module`-derived class to define an
# adjustable kernel
# %%
# We also define a filter with three smearing parameters, corresponding to
# three channels


class ParamKernel(torch.nn.Module):
    def __init__(self, sigma: torch.Tensor, a0: torch.Tensor):
        super(ParamKernel, self).__init__()
        self._sigma = sigma
        self._a0 = a0

    def forward(self, k2):

        filter = torch.stack([torch.exp(-k2 * s**2 / 2) for s in self._sigma])
        filter[0, :] *= self._a0[0] / (1 + k2)
        filter[1, :] *= self._a0[1] / (1 + k2**3)
        return filter


# %%
# This can be used just as a simple filter

sigma = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
a0 = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
sigma.requires_grad_(True)
a0.requires_grad_(True)

kernel = ParamKernel(sigma, a0)
KF = torchpme.lib.KSpaceFilter(cell, ns, kernel=kernel)

filtered = KF.compute(mesh)

filtered_at_positions = MI.mesh_to_points(filtered)

# %%
#

value = (charges * filtered_at_positions).sum()

# %%
print(value)


# %%
value.backward()

# %%

charges.grad


# %%

sigma.grad
# %%
a0.grad
# %%
