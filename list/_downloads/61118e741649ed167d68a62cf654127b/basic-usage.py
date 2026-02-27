"""
Basic Usage
===========

.. currentmodule:: torchpme

This example showcases how the main capabilities of ``torchpme``.
We build a simple ionic crystal and compute the electrostatic potential and the forces
for each atom.

To follow this tutorial, it is assumed that torch-pme has been :ref:`installed
<userdoc-installation>` on your computer.

.. note::

    Inside this tutorial and all other examples of the documentation you can click on
    each object in a code block to be forwarded to the documentation of the class or
    function to get further details.

"""

# %%

import chemiscope
import torch
from ase import Atoms
from vesin.torch import NeighborList

from torchpme import EwaldCalculator
from torchpme.potentials import CoulombPotential

# %%
#
# We initially set the ``device`` and ``dtype`` for the calculations. We will use the
# CPU for this and double precision. If you have a CUDA devide you can also set the
# device to ``"cuda"`` to use the GPU and speed up the calculations. Double precision is
# a requirement for the neighbor list implementation that we are using here.

device = "cpu"
dtype = torch.float64


# %%
#
# Generate Simple Example Structures
# ----------------------------------
#
# Throughout this tutorial, we will work with a simple atomic structure in three
# dimensions, which is a distorted version of the CsCl structure. The goal will be to
# compute the electrostatic potentials and the forces for each atom.
#
# We first generate a single unit cell of CsCl (2 atoms in the cell) where the Cs atoms
# get a charge of +1 and the Cl atom a charge of -1.

atoms_unitcell = Atoms(
    "CsCl", cell=torch.eye(3), positions=[[0, 0, 0], [0.5, 0.5, 0.5]]
)
atoms_unitcell.set_initial_charges([1, -1])

# %%
#
# We next generate a bigger structure by periodically repeating the unit cell 3 times in
# each direction and apply a small random distortion to all atoms.

atoms = atoms_unitcell.repeat(3)
atoms.rattle(stdev=0.01)
atoms.wrap()  # make sure all atoms are in the unit cell

chemiscope.show(
    frames=[atoms],
    mode="structure",
    settings=chemiscope.quick_settings(structure_settings={"unitCell": True}),
)

# %%
#
# We now extract the required properties from the :class:`ase.Atoms` object and store
# them as individial variables as :class:`torch.Tensor`, which is the required input
# type for *torch-pme*.
#
# For the positions, we explicitly set `requires_grad=True`. This is because we are
# ultimately interested in computing the forces, which are the gradients of the total
# (electrostatic) energy with respect to the positions (up to a minus sign).
# *torch-pme* can automatically compute such gradients, for which we need to communicate
# at this point that we will need to take gradients with respect to the positions in
# the future.

positions = torch.tensor(
    atoms.positions, dtype=dtype, device=device, requires_grad=True
)
cell = torch.tensor(atoms.cell.array, dtype=dtype, device=device)
charges = torch.tensor(
    atoms.get_initial_charges(), dtype=dtype, device=device
).unsqueeze(1)

# %%
#
# Tuning: Find Optimal Hyperparameters
# ------------------------------------
#
# Ewald and mesh methods require the specification of multiple hyperparameters, namely
#
# 1. the cutoff radius :math:`r_\mathrm{cut}`` for the short-range parts
# 2. the ``smearing`` parameter $\sigma$ determining the relative importance of the
#    short-range and long-range terms in the split
# 3. either the mesh spacing :math:`h` for mesh-based methods, or a reciprocal space
#    cutoff :math:`k_\mathrm{cut} = 2\pi/\lambda`` for the Ewald sum, where
#    :math:`\lambda` is the shortest wavelength used in the Fourier series and
#    corresponds to :math:`h` for mesh-based approaches
#
# For ML applications, we typically first select a short-range cutoff to be the same to
# conventional short-ranged ML models, and define the remaining parameters from there.
# In this example, we are simply computing the Coulomb potential, and thus compute the
# hyperparameters simply based on convergence criteria.

box_length = cell[0, 0]
rcut = float(box_length) / 2 - 1e-10
smearing = rcut / 5
# lambda which gives the reciprocal space cutoff kcut=2*pi/lambda
lr_wavelength = smearing / 2

# %%
#
# However, especially for users without much experience on how to choose these
# hyperparameters, we have built-in tuning functions for each Calculator (see below)
# such as :func:`torchpme.tuning.tune_ewald`, which can automatically find a good set
# of parameters. These can be used like this:
#
# .. code-block:: python
#
#   sum_charges_sq = torch.sum(charges**2, dim=0)
#   smearing, lr_wavelength, rcut = tune_ewald(sum_charges_sq, cell, positions, accuracy=1e-1)
#
# Define Potential
# ----------------
#
# We now need to define the potential function with which the atoms interact. Since this
# is a library for long-range ML, we support three major options:
#
# 1. the :class:`Coulomb potential <CoulombPotential>` (:math:`1/r`)
# 2. more general :class:`inverse power-law potentials <InversePowerLawPotential>` (:math:`1/r^p`)
# 3. an option to build custom potentials using :class:`splines <SplinePotential>`
#
# This tutorial focuses on option (1), which is the most relevant from a practical point
# of view. We can simply initialize an instance of the :class:`CoulombPotential` class that
# contains all the necessary functions (such as those defining the short-range and
# long-range splits) for this potential and makes them useable in the rest of the code.

potential = CoulombPotential(smearing=smearing, device=device, dtype=dtype)

# %%
#
# Neighbor List
# -------------
#
# *torch-pme* requires us to compute the neighbor list (NL) in advance. This is because
# for many ML applications, we would otherwise need to repeat this computation multiple
# times during model training of a neural network etc. By computing it externally and
# providing it as an input, we can streamline training workflows.
# We compute the neighbor list using the ``vesin`` package, which also provides a
# pytorch implementation that retains the computational graph. This will later allow us
# to automatically compute gradients of the distances with respect to the atomic
# coordinates. More details can be found in its documentation.

nl = NeighborList(cutoff=rcut, full_list=False)
neighbor_indices, neighbor_distances = nl.compute(
    points=positions, box=cell, periodic=True, quantities="Pd"
)

# %%
#
# Main Part: Calculator
# ---------------------
#
# The ``Calculator`` classes are the main user-facing classes in *torch-pme*. These are
# used to compute atomic potentials :math:`V_i`` for a given set of positions and
# particle weights (charges). For periodic calculators that are the main focus of this
# tutorial, it is also required to specify a ``cell``. We have three periodic
# calculators:
#
# 1. :class:`EwaldCalculator`: uses the Ewald sum. Recommended for structures with
#    :math:`<10000` atoms due to its high accuracy and simplicity. Should be avoided for
#    big structures due to the slower :math:`\mathcal{O}(N^2)` to
#    :math:`\mathcal{O}(N^{\frac{3}{2}})` scaling of the computational cost (depending
#    on how the hyperparameters are chosen) with respect to the number of atoms.
# 2. :class:`PMECalculator`: uses the Particle Mesh Ewald (PME) method. Mostly for
#    reference.
# 3. :class:`P3MCalculator`: uses the Particle-Particle Particle-Mesh (P3M) method.
#    Recommended for structures :math:`\ge 10000` atoms due to the efficient
#    :math:`\mathcal{O}(N\log N)` scaling of the computational cost with the number of
#    atoms.
#
# Since our structure is relatively small, we use the :class:`EwaldCalculator`.
# We start by the initialization of the class.

calculator = EwaldCalculator(
    potential=potential, lr_wavelength=lr_wavelength, device=device, dtype=dtype
)

# %%
#
# Compute Energy
# --------------
#
# We have now all ingredients: we can use the ``Calculator`` class to, well, actually
# compute the potentials :math:`V_i` at the position of the atoms, or the total energy
# for the given particle weights (``charges``). The electrostatic potential can then be
# obtained as
#
# .. math::
#
#   E = \sum_{i=1}^N q_i V_i

potentials = calculator.forward(
    charges, cell, positions, neighbor_indices, neighbor_distances
)
energy = torch.sum(charges * potentials)

print("Energy = \n", energy.item())

# %%
#
# .. hint::
#
#   The energy is in Gaussian units. You can change the unit system by setting the
#   ``prefactor`` when initializing the calculator. For more details about this refer to
#   :ref:`prefactors`.
#
# Compute Forces using backpropagation (automatic differentiation)
# ----------------------------------------------------------------
#
# The forces on the particles can simply be obtained as minus the gradient of the energy
# with respect to the positions. These are easy to evaluate using the automatic
# differentiation capabilities of `pytorch
# <https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html>`_ using the
# backpropagation method.
#
# Note that this only works since we set ``requires_grad=True`` when we initialized the
# positions tensor above.

energy.backward()
forces = -positions.grad

print("Force on first atom = \n", forces[0])

# %%
#
# Aperiodic Structures
# --------------------
#
# For now, we have been using the :class:`EwaldCalculator` which is a periodic
# calculator. We can however also use it for aperiodic structures by just using it as a
# calculator with a cutoff radius. To start clone the positions to avoid accumulating
# gradients with respect to the same variables multiple times

positions_aperiodic = positions.clone().detach()
positions_aperiodic.requires_grad = True

# %%
#
# Compute neighbor list but this time without periodic boudary conditions

neighbor_indices_aperiodic, neighbor_distances_aperiodic = nl.compute(
    points=positions_aperiodic, box=cell, periodic=False, quantities="Pd"
)

# %%
#
# Compute aperiodic potential using the dedicated subroutine that is present
# in all calculators. For now, we do not provide an explicit calculator for aperiodic
# systems since the focus in mainly on periodic ones.

potentials_aperiodic = calculator._compute_rspace(
    charges, neighbor_indices_aperiodic, neighbor_distances_aperiodic
)

# %%
#
# Compute total energy and forces

energy_aperiodic = torch.sum(charges * potentials_aperiodic)
energy_aperiodic.backward()
forces_aperiodic = positions_aperiodic.grad  # forces on the particles

# %%
#
# References to more advanced tasks
# ---------------------------------
#
# Refer to :ref:`examples <userdoc-how-to>` for tackling specific and advanced tasks and
# to the :ref:`references <userdoc-reference>` for more information the spefific API.
