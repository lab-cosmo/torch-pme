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

from time import time

import ase
import torch

import torchpme

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

ns = torch.tensor([5, 5, 5])
interpolator = torchpme.lib.MeshInterpolator(cell=cell, ns_mesh=ns, order=3)
interpolator.compute_weights(positions)
mesh = interpolator.points_to_mesh(charges)

value = mesh.sum()

# %%
# The gradients can be computed by just running `backward` on the
# end result.
# Because of the sum rules that apply to the interpolation scheme,
# the gradients with respect to positions and cell entries are zero,
# and the gradients relative to the charges are all 1.

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
# function of other quantities. For instance, pretend
# there is an external electric field along :math:`x`,
# and that the weights should be proportional to the
# electrostatic energy at each atom position
# (NB: defining an electric field in a periodic setting is
# not so simple, this is just a toy example).

positions.grad.zero_()
charges.grad.zero_()
cell.grad.zero_()

weights = charges * positions[:, :1]
mesh3 = interpolator.points_to_mesh(weights)

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
# can also be differentiated through.

# %%
# A parametric k-space filter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We define a filter with multiple smearing parameters,
# that are applied separately to multiple mesh channels


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
# We define a 2D weights (to get a 2D mesh), and
# define parameters as optimizable quantities

weights = torch.tensor(
    [
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ],
    dtype=dtype,
    device=device,
)

torch.autograd.set_detect_anomaly(True)
sigma = torch.tensor([1.0, 0.5], dtype=dtype, device=device)
a0 = torch.tensor([1.0, 2.0], dtype=dtype, device=device)

positions = positions.detach()
cell = cell.detach()
positions.requires_grad_(True)
cell.requires_grad_(True)

weights = weights.detach()
sigma = sigma.detach()
a0 = a0.detach()
weights.requires_grad_(True)
sigma.requires_grad_(True)
a0.requires_grad_(True)

# %%
# Compute the mesh, apply the filter, and also complete the
# PME-like operation by evaluating the transformed mesh
# at the atom positions

interpolator = torchpme.lib.MeshInterpolator(cell, ns, 3)
interpolator.compute_weights(positions)
mesh = interpolator.points_to_mesh(weights)

kernel = ParametricKernel(sigma, a0)
KF = torchpme.lib.KSpaceFilter(cell, ns, kernel=kernel)

filtered = KF.compute(mesh)

filtered_at_positions = interpolator.mesh_to_points(filtered)

# %%
# Computes a (rather arbitrary) function of the outputs,
# backpropagates and then outputs the gradients.
# With this messy non-linear function everything has
# nonzero gradients

value = (charges * filtered_at_positions).sum()
value.backward()

# %%
print(
    f"""
Value: {value}

Position gradients:
{positions.grad.T}

Cell gradients:
{cell.grad}

Weights gradients:
{weights.grad.T}

Param. a0:
{a0.grad}

Param. sigma:
{sigma.grad}
"""
)

# %%
# A ``torch`` module based on ``torchpme``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# It is also possible to combine all this in a
# custom :py:class:`torch.nn.Module`, which is the
# first step towards designing a model training pipeline
# based on a custom ``torchpme`` model.

# %%
# We start by defining a Yukawa-like potential, and
# a (rather contrieved) model that combines a Fourier
# filter, with a multi-layer perceptron to post-process
# charges and "potential".


# Define the kernel
class SmearedCoulomb(torchpme.lib.KSpaceKernel):
    def __init__(self, sigma2):
        super().__init__()
        self._sigma2 = sigma2

    def from_k_sq(self, k2):
        # we use a mask to set to zero the Gamma-point filter
        mask = torch.ones_like(k2, dtype=torch.bool, device=k2.device)
        mask[..., 0, 0, 0] = False
        potential = torch.zeros_like(k2)
        potential[mask] = torch.exp(-k2[mask] * self._sigma2 * 0.5) / k2[mask]
        return potential


# Define the module
class KSpaceModule(torch.nn.Module):
    """A demonstrative model combining torchpme and a multi-layer perceptron"""

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
            cell=dummy_cell, ns_mesh=torch.tensor([1, 1, 1]), order=3
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
        # use a helper function to get the mesh size given resolution
        ns_mesh = torchpme.lib.get_ns_mesh(cell, self._mesh_spacing)
        ns_mesh = torch.tensor([4, 4, 4])
        self._interpolator = torchpme.lib.MeshInterpolator(
            cell=cell, ns_mesh=ns_mesh, order=3
        )
        self._interpolator.compute_weights(positions)
        mesh = self._interpolator.points_to_mesh(charges)

        self._KF.update_mesh(cell, ns_mesh)
        self._KF.update_filter()
        mesh = self._KF.compute(mesh)
        pot = self._interpolator.mesh_to_points(mesh)

        x = torch.hstack([charges, pot])
        for layer in self._layers:
            x = layer(x)
        # Output layer
        x = self._output_layer(x)
        return x.sum()


# %%
# Creates an instance of the model and evaluates it.

my_module = KSpaceModule(sigma2=1.0, mesh_spacing=1.0, hidden_sizes=[10, 4, 10])

# (re-)initialize vectors

charges = charges.detach()
positions = positions.detach()
cell = cell.detach()
charges.requires_grad_(True)
positions.requires_grad_(True)
cell.requires_grad_(True)

value = my_module.forward(positions, cell, charges)
value.backward()


# %%
# Gradients compute, and look reasonable!

print(
    f"""
Value: {value}

Position gradients:
{positions.grad.T}

Cell gradients:
{cell.grad}

Charges gradients:
{charges.grad.T}
"""
)

# %%
# ... also on the MLP parameters!

for layer in my_module._layers:
    print(layer._parameters)

# %%
# It's always good to run some `gradcheck`...

my_module.zero_grad()
check = torch.autograd.gradcheck(
    my_module,
    (
        torch.randn((16, 3), device=device, dtype=dtype, requires_grad=True),
        torch.randn((3, 3), device=device, dtype=dtype, requires_grad=True),
        torch.randn((16, 1), device=device, dtype=dtype, requires_grad=True),
    ),
)
if check:
    print("gradcheck passed for custom torch-pme module")
else:
    raise ValueError("gradcheck failed for custom torch-pme module")


# %%
# Jitting a custom module
# ~~~~~~~~~~~~~~~~~~~~~~~
# The custom module can also be jitted!

old_cell_grad = cell.grad.clone()
jit_module = torch.jit.script(my_module)

jit_charges = charges.detach()
jit_positions = positions.detach()
jit_cell = cell.detach()
jit_cell.requires_grad_(True)
jit_charges.requires_grad_(True)
jit_positions.requires_grad_(True)

jit_value = jit_module.forward(jit_positions, jit_cell, jit_charges)
jit_value.backward()

# %%
# Values match within machine precision

print(
    f"""
Delta-Value: {value-jit_value}

Delta-Position gradients:
{positions.grad.T-jit_positions.grad.T}

Delta-Cell gradients:
{cell.grad-jit_cell.grad}

Delta-Charges gradients:
{charges.grad.T-jit_charges.grad.T}
"""
)

# %%
# We can also time the difference in execution
# time between the Pytorch and scripted versions of the
# module (depending on the system, the relative efficiency
# of the two evaluations could go either way!)

duration = 0.0
for _i in range(20):
    my_module.zero_grad()
    positions = positions.detach()
    cell = cell.detach()
    charges = charges.detach()
    duration -= time()
    value = my_module.forward(positions, cell, charges)
    value.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    duration += time()
time_python = (duration) * 1e3 / 20

duration = 0.0
for _i in range(20):
    jit_module.zero_grad()
    positions = positions.detach()
    cell = cell.detach()
    charges = charges.detach()
    duration -= time()
    value = jit_module.forward(positions, cell, charges)
    value.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    duration += time()
time_jit = (duration) * 1e3 / 20


# %%
print(f"Evaluation time:\nPytorch: {time_python}ms\nJitted:  {time_jit}ms")

# %%
