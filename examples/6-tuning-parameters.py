"""
.. _example-tuning-parameters:

Tune parameters of ``Calculators``
==================================

:Authors: Qianjun Xu `@GardevoirX <https://github.com/GardevoirX>`_

This example shows how to use the tuning function :py:func:`tune_pme` to select the
parameters of :py:class:`PMECalculator` with respect to the energy accuracy
requirements. Similarly, one can use :py:func:`tune_ewald` to do the same thing for
:py:class:`EwaldCalculator`.
"""

import torch
from vesin.torch import NeighborList

from torchpme import CoulombPotential, PMECalculator
from torchpme.utils import tune_pme

device = "cpu"
dtype = torch.float64
rng = torch.Generator()
rng.manual_seed(32)

# %%
# We start by generating a simple CsCl crystal structure. The reference madelung
# constant is 2.035361.

positions = torch.tensor(
    [[0.0000, 0.0000, 0.0000], [0.5000, 0.5000, 0.5000]], device=device, dtype=dtype
)
charges = torch.tensor([[-1.0], [1.0]], device=device, dtype=dtype)
cell = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
)

madelung_ref = 2.035361

# %%
# `tune_pme` takes the sum of squared charges, cell and atom positions as the input.
# One can also specify the `interpolation_nodes` and the exponent in the
# :math:`1/r^p` potential if the default 4 interpolation nodes and :math:`1/r`
# potential is not used.
# Because the tuning method is gradient-based optimization, we provide the user with # an interface tuning the learning rate and maximum steps for
# :py:class:`Adam <torch.optim.Adam>` optimizer.
# The default learning rate is set small to make the optimization stable
# for small systems (<100 atoms), so if the system is large, one can consider a
# larger learning rate to accelerate the optimization.

smearing, params, cutoff = tune_pme(
    torch.sum(charges**2, dim=0), cell, positions, accuracy=1e-6
)
print(params)

# %%
# With the parameters, we can do the neighbor list calculation and set up the
# potential and the calculator.

nl = NeighborList(cutoff=cutoff, full_list=False)
i, j, neighbor_distances = nl.compute(
    points=positions, box=cell, periodic=True, quantities="ijd"
)
neighbor_indices = torch.stack([i, j], dim=1)

potential = CoulombPotential(smearing=smearing)
calculator = PMECalculator(
    potential=potential,
    **params,
)
potential = calculator.forward(
    positions=positions,
    charges=charges,
    cell=cell,
    neighbor_indices=neighbor_indices,
    neighbor_distances=neighbor_distances,
)

# %%
# Now we can compare our result with the reference. The relative error is smaller
# than the default accuracy :math:`10^{-6}`.

print(
    f"rel_err = {torch.abs((-torch.sum(potential * charges) - madelung_ref) / madelung_ref)}"
)
# %%
