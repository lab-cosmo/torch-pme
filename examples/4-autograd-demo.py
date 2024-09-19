"""
.. _example-autograd-demo:

Custom ``torch-pme`` models with automatic differentiation
==========================================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This example showcases how the main building blocks of ``torchpme``,
:py:class:`MeshInterpolator` and :py:class:`KSpacaFilter` can be
combined creatively to construct arbitrary models that incorporate
long-range structural correlations.

None of the models presented here has probably much meaning, and
the use in a ML setting (including the definition of an appropriate
loss, and its optimization) is left as an exercise to the reader.
"""

# %%
# Import dependencies

import ase

# import chemiscope
import torch

import torchpme


# from matplotlib import pyplot as plt

device = "cpu"
dtype = torch.float64
rng = torch.Generator()
rng.manual_seed(32)

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
# Autodifferentiation through the core ``torchpme`` classes
# ---------------------------------------------------------
# We begin by showing how it is possible to compute a function of the internal state
# for the core classes, and to differentiate with respect to the structural and input
# parameters.

# %%
# Functions of the atom density
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The construction of a "decorated atom density" through
# :py:class:`MeshInterpolator <torchpme.lib.MeshInterpolator>`
# can be easily differentiated through.
# We only need to request a gradient evaluation, evaluate the grid, and compute
# a function of the grid points (again, this is a proof-of-principle example,
# probably not very useful in practice).

positions.requires_grad_(True)
charges.requires_grad_(True)
cell.requires_grad_(True)
positions.grad.zero_()
charges.grad.zero_()
cell.grad.zero_()

ns = torch.tensor([5, 5, 5])
MI = torchpme.lib.MeshInterpolator(
    cell=cell,
    ns_mesh=ns,
    interpolation_order=3,
)
MI.compute_interpolation_weights(positions)
mesh = MI.points_to_mesh(charges)

value = mesh.sum()

# %%
# The gradients can be computed by just running `backward` on the
# end result.
# Because of the sum rules that apply to the interpolation scheme,
# the gradients with respect to positions and cell entries are zero,
# and the gradients relative to the charges are all 1

# we keep the graph to compute another quantity
value.backward(retain_graph=True)

print(
    f"""
Position gradients:
{positions.grad.T}

Cell gradients:
{cell.grad}

Charges gradients:
{charges.grad.T}
"""
)

# %%
# If we apply a non-linear function before summing,
# these sum rules apply only approximately.

positions.grad.zero_()
charges.grad.zero_()
cell.grad.zero_()

value2 = torch.sin(mesh).sum()
value2.backward(retain_graph=True)

print(
    f"""
Position gradients:
{positions.grad.T}

Cell gradients:
{cell.grad}

Charges gradients:
{charges.grad.T}
"""
)

# %%
# Indirect functions of the weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# It is possible to have the atomic weights be a
# function of other quantities, e.g. in a way pretend
# there is an external electric field along :math:`x`
# (NB defining an electric field in a periodic setting is
# not so simple, this is just a toy example)

positions.grad.zero_()
charges.grad.zero_()
cell.grad.zero_()

weights = charges * positions[:, :1]
mesh3 = MI.points_to_mesh(weights)

value3 = mesh3.sum()
value3.backward()

print(
    f"""
Position gradients:
{positions.grad.T}

Cell gradients:
{cell.grad}

Charges gradients:
{charges.grad.T}
"""
)

# %%
# Optimizable k-space filter
# --------------------------
# The operations in a
# :py:class:`KSpaceFilter <torchpme.lib.KSpaceFilter>`
# can also be differentiated through


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


class ParametricKernel(torch.nn.Module):
    def __init__(self, sigma: torch.Tensor, a0: torch.Tensor):
        super().__init__()
        self._sigma = sigma
        self._a0 = a0

    def from_k_sq(self, k2):

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

kernel = ParametricKernel(sigma, a0)
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
cell.grad

# %%
# A ``torch`` module based on ``torchpme``
# ----------------------------------------
#


class SmearedCoulomb(torchpme.lib.KSpaceKernel):
    def __init__(self, sigma2):
        super().__init__()
        self._sigma2 = sigma2

    def from_k_sq(self, k2):
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
        self, mesh_spacing: float = 0.5, sigma2: float = 1.0, hidden_sizes=None
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
            cell=dummy_cell,
            ns_mesh=torch.tensor([1, 1, 1]),
            kernel=SmearedCoulomb(self._sigma2),
        )

        if hidden_sizes is None:  # default architecture
            hidden_sizes = [10, 10]
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

        x = torch.hstack([charges, pot])
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
# this can also be jitted!
old_cell_grad = cell.grad
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
cell.grad - old_cell_grad
# %%
