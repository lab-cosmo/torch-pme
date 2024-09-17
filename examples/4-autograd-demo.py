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
cell.requires_grad_(True)

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
        super().__init__()
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
print(cell.grad)

# %%
# A ``torch`` module based on ``torchpme``
# ----------------------------------------
#

class SmearedCoulomb(torch.nn.Module):
    def __init__(self, sigma2):
        super().__init__()
        self._sigma2=sigma2

    def forward(self, k2):
        mask = torch.ones_like(k2, dtype=torch.bool, device=k2.device)
        mask[..., 0, 0, 0] = False
        potential = torch.zeros_like(k2)
        potential[mask] = torch.exp(-k2[mask] * self._sigma2 * 0.5) / k2[mask]
        return potential

class KSpaceModule(torch.nn.Module):
    """
    A 
    """
    def __init__(
        self, mesh_spacing: float = 0.5, sigma2: float = 1.0, hidden_sizes=[10, 10]
    ):
        super().__init__()
        self._mesh_spacing = mesh_spacing

        # degree of smearing as an optimizable parameter
        self._sigma2 = torch.nn.Parameter(
            torch.tensor(sigma2, dtype=dtype, device=device)
        )

        dummy_cell = torch.eye(3, dtype=dtype)
        self._MI = torchpme.lib.MeshInterpolator(
            cell=dummy_cell,
            ns_mesh=torch.tensor([1, 1, 1]),
            interpolation_order=3,
        )
        self._KF = torchpme.lib.KSpaceFilter(
            cell=dummy_cell, ns_mesh=torch.tensor([1, 1, 1]), 
            kernel=SmearedCoulomb(self._sigma2)
        )

        # a neural network to process "charge and potential"
        last_size = 2  # input is charge and potential
        self._layers = torch.nn.ModuleList()
        for hidden_size in hidden_sizes:
            self._layers.append(
                torch.nn.Linear(last_size, hidden_size, dtype=dtype, device=device)
            )
            self._layers.append(torch.nn.Tanh())
            last_size = hidden_size
        self._output_layer = torch.nn.Linear(
            last_size, 1, dtype=dtype, device=device
        )  # outputs one value

    def forward(self, positions, cell, charges):

        # ns_mesh = torchpme.lib.get_ns_mesh(cell, self._mesh_spacing)
        ns_mesh = torch.tensor([4, 4, 4])
        self._MI = torchpme.lib.MeshInterpolator(
            cell=cell,
            ns_mesh=ns_mesh,
            interpolation_order=3,
        )
        self._MI.compute_interpolation_weights(positions)
        mesh = self._MI.points_to_mesh(charges)

        self._KF.update_mesh(cell, ns_mesh)
        self._KF.update_filter()
        mesh = self._KF.compute(mesh)
        pot = self._MI.mesh_to_points(mesh)
        return (pot * charges).sum()
        x = torch.vstack([charges, pot])
        for layer in self._layers:
            x = layer(x)
        # Output layer
        x = self._output_layer(x)
        return x.sum()


# %%

my_module = KSpaceModule(sigma2=1, mesh_spacing=1, hidden_sizes=[10, 4, 10])

# %%
if charges.grad is not None:
    charges.grad.zero_()
if positions.grad is not None:
    positions.grad.zero_()
if cell.grad is not None:
    cell.grad.zero_()

positions.requires_grad_(True)
cell.requires_grad_(True)
value = my_module.forward(positions, cell, charges)
# %%
torch.autograd.set_detect_anomaly(True)
value.backward()

# %%
charges.grad
# %%
positions.grad
# %%
cell.grad
# %%
my_tsmod = torch.jit.script(my_module)

# %%
if charges.grad is not None:
    charges.grad.zero_()
if positions.grad is not None:
    positions.grad.zero_()
if cell.grad is not None:
    cell.grad.zero_()

positions.requires_grad_(True)
cell.requires_grad_(True)
value = my_tsmod.forward(positions, cell, charges)

# %%
value.backward()

# %%
cell.grad
# %%
