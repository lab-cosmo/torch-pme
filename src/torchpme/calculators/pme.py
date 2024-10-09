import math
import warnings
from typing import Literal, Optional

import torch
from torch import profiler

from ..lib import Potential
from ..lib.kspace_filter import KSpaceFilter
from ..lib.kvectors import get_ns_mesh, get_ns_mesh_differentiable
from ..lib.mesh_interpolator import MeshInterpolator, compute_RMS_phi
from ..lib.potentials import gamma
from .base import Calculator, estimate_smearing

class PMECalculator(Calculator):
    r"""
    Potential using a particle mesh-based Ewald (PME).

    Scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N` used as a reference to test faster implementations.

    For computing a **neighborlist** a reasonable ``cutoff`` is half the length of
    the shortest cell vector, which can be for example computed according as

    .. code-block:: python

        cell_dimensions = torch.linalg.norm(cell, dim=1)
        cutoff = torch.min(cell_dimensions) / 2 - 1e-6

    This ensures a accuracy of the short range part of ``1e-5``.

    :param potential: A :py:class:`Potential` object that implements the evaluation
        of short and long-range potential terms. The ``smearing`` parameter
        of the potential determines the split between real and k-space regions.
        For a :py:class:`torchpme.lib.CoulombPotential` it corresponds
        to the smearing of the atom-centered Gaussian used to split the
        Coulomb potential into the short- and long-range parts. A reasonable value for
        most systems is to set it to ``1/5`` times the neighbor list cutoff.
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically be set
        to half of ``smearing``.
    :param interpolation_nodes: The number ``n`` of nodes used in the interpolation per
        coordinate axis. The total number of interpolation nodes in 3D will be ``n^3``.
        In general, for ``n`` nodes, the interpolation will be performed by piecewise
        polynomials of degree ``n - 1`` (e.g. ``n = 4`` for cubic interpolation).
        Only the values ``3, 4, 5, 6, 7`` are supported.
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.


    For an **example** on the usage for any calculator refer to :ref:`userdoc-how-to`.
    """

    def __init__(
        self,
        potential: Potential,
        mesh_spacing: Optional[float] = None,
        interpolation_nodes: int = 4,
        full_neighbor_list: bool = False,
    ):
        super().__init__(
            potential=potential,
            full_neighbor_list=full_neighbor_list,
        )

        if potential.smearing is None:
            raise ValueError(
                "Must specify range radius to use a potential with EwaldCalculator"
            )

        if mesh_spacing is None:
            mesh_spacing = potential.smearing / 8.0
        self.mesh_spacing: float = mesh_spacing

        if interpolation_nodes not in [3, 4, 5, 6, 7]:
            raise ValueError("Only `interpolation_nodes` from 3 to 7 are allowed")
        self.interpolation_nodes: int = interpolation_nodes

        # Initialize the filter module. Set dummy value for smearing to propper
        # initilize the `KSpaceFilter` below
        self._KF: KSpaceFilter = KSpaceFilter(
            cell=torch.eye(3),
            ns_mesh=torch.ones(3, dtype=int),
            kernel=self.potential,
            fft_norm="backward",
            ifft_norm="forward",
        )

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: Kernel function `G` and initialization of `MeshInterpolator` only depend
        # on `cell`. Caching may save up to 15% but issues with AD should be taken
        # resolved.

        with profiler.record_function("init 0: preparation"):
            # Compute number of times each basis vector of the reciprocal space can be
            # scaled until the cutoff is reached
            ns = get_ns_mesh(cell, self.mesh_spacing)

        with profiler.record_function("init 1: initialize mesh interpolator"):
            interpolator = MeshInterpolator(
                cell,
                ns,
                interpolation_nodes=self.interpolation_nodes,
                method="Lagrange",  # convention for classic PME
            )

        with profiler.record_function("update the mesh for the k-space filter"):
            self._KF.update_mesh(cell, ns)

        with profiler.record_function("step 1: compute density interpolation"):
            interpolator.compute_weights(positions)
            rho_mesh = interpolator.points_to_mesh(particle_weights=charges)

        with profiler.record_function("step 2: perform actual convolution using FFT"):
            potential_mesh = self._KF.compute(rho_mesh)

        with profiler.record_function("step 3: back interpolation + volume scaling"):
            ivolume = torch.abs(cell.det()).pow(-1)
            interpolated_potential = (
                interpolator.mesh_to_points(potential_mesh) * ivolume
            )

        with profiler.record_function("step 4: remove the self-contribution"):
            # Using the Coulomb potential as an example, this is the potential generated
            # at the origin by the fictituous Gaussian charge density in order to split
            # the potential into a SR and LR part. This contribution always should be
            # subtracted since it depends on the smearing parameter, which is purely a
            # convergence parameter.
            fill_value = self.potential.self_contribution()
            self_contrib = torch.full([], fill_value, device=self._device)
            interpolated_potential -= charges * self_contrib

        with profiler.record_function("step 5: charge neutralization"):
            # If the cell has a net charge (i.e. if sum(charges) != 0), the method
            # implicitly assumes that a homogeneous background charge of the opposite
            # sign is present to make the cell neutral. In this case, the potential has
            # to be adjusted to compensate for this. An extra factor of 2 is added to
            # compensate for the division by 2 later on
            charge_tot = torch.sum(charges, dim=0)
            prefac = self.potential.background_correction()
            interpolated_potential -= 2 * prefac * charge_tot * ivolume

        # Compensate for double counting of pairs (i,j) and (j,i)
        return interpolated_potential / 2


def tune_pme(
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        interpolation_nodes: int,
        exponent: int = 1,
        accuracy: Optional[Literal["medium", "accurate"] | float] = "medium",
        max_steps: int = 50000,
        learning_rate: float = 1e-2,
        verbose: bool = False,
):
    r"""Find the optimal parameters for a single system for the ewald method.

    For the error formulas are given `elsewhere <https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/4/4d/Script_Longrange_Interactions.pdf>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha &= \left( \sqrt{2}\,\mathrm{smearing} \right)^{-1}

        K &= \frac{2 \pi}{\mathrm{lr\_wavelength}}

        r_c &= \mathrm{cutoff}
    """

    if exponent != 1:
        raise NotImplementedError("Only exponent = 1 is supported")
    dtype = positions.dtype
    device = positions.device

    # Create valid dummy tensors to verify `positions`, `charges` and `cell`
    neighbor_indices = torch.zeros(0, 2, device=device)
    neighbor_distances = torch.zeros(0, device=device)
    Calculator._validate_compute_parameters(
        positions=positions,
        charges=charges,
        cell=cell,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )

    positions = positions[0]
    charges = charges[0]
    cell = cell[0]

    if charges.shape[1] > 1:
        raise NotImplementedError(
            f"Found {charges.shape[1]} charge channels, but only one iss supported"
        )

    if accuracy == "medium":
        accuracy = 1e-3
    elif accuracy == "accurate":
        accuracy = 1e-6
    elif not isinstance(accuracy, float):
        raise ValueError(
            f"'{accuracy}' is not a valid method or a float: Choose from 'fast',"
            f"'medium' or 'accurate', or provide a float for the accuracy."
        )

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = float(torch.min(cell_dimensions) / 2)
    
    smearing_init = estimate_smearing(cell)
    prefac = 2 * torch.sum(charges**2) / math.sqrt(len(positions))
    volume = torch.abs(cell.det())
    interpolation_nodes = torch.tensor(interpolation_nodes)

    def smooth_mesh_spacing(mesh_spacing):
        """Confine to (0, min_dimension), ensuring that the ``ns``
        parameter is not smaller than 1
        (see :py:func:`_compute_lr` of :py:class:`PMEPotential`)."""
        return min_dimension * torch.sigmoid(mesh_spacing)

    def err_Fourier(smearing, mesh_spacing):
        def H(mesh_spacing):
            return torch.prod(1 / get_ns_mesh_differentiable(cell, mesh_spacing)) ** (
                1 / 3
            )

        def RMS_pi(mesh_spacing):
            ns_mesh = get_ns_mesh_differentiable(cell, mesh_spacing)
            # print(torch.linalg.norm(interpolator.compute_RMS_phi(ns_mesh, positions)))
            return torch.linalg.norm(
                compute_RMS_phi(cell, interpolation_nodes, ns_mesh, positions)
            )

        def log_factorial(x):
            return torch.lgamma(x + 1)

        def factorial(x):
            return torch.exp(log_factorial(x))

        return (
            prefac
            * torch.pi**0.25
            * (6 * (1 / 2**0.5 / smearing) / (2 * interpolation_nodes + 1)) ** 0.5
            / volume ** (2 / 3)
            * (2**0.5 / smearing * H(mesh_spacing)) ** interpolation_nodes
            / factorial(interpolation_nodes)
            * torch.exp(
                (interpolation_nodes) * (torch.log(interpolation_nodes / 2) - 1) / 2
            )
            * RMS_pi(mesh_spacing)
        )

    def err_real(smearing, cutoff):
        return (
            prefac
            / torch.sqrt(cutoff * volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def loss(smearing, mesh_spacing, cutoff):
        return torch.sqrt(
            err_Fourier(smearing, mesh_spacing) ** 2 + err_real(smearing, cutoff) ** 2
        )

    # initial guess
    smearing = torch.tensor(
        smearing_init, device=device, dtype=dtype, requires_grad=True
    )
    mesh_spacing = torch.tensor(
        -math.log(min_dimension * 8 / smearing_init - 1),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )  # smooth_mesh_spacing(mesh_spacing) = smearing / 8, which is the standard initial guess
    cutoff = torch.tensor(half_cell / 5, device=device, dtype=dtype, requires_grad=True)

    optimizer = torch.optim.Adam([smearing, mesh_spacing, cutoff], lr=learning_rate)

    for step in range(max_steps):
        loss_value = loss(smearing, smooth_mesh_spacing(mesh_spacing), cutoff)
        if torch.isnan(loss_value):
            raise ValueError(
                "The value of the estimated error is now nan, consider using a "
                "smaller learning rate."
            )
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose and (step % 100 == 0):
            print(f"{step}: {loss_value:.2e}")
        if loss_value <= accuracy:
            break

    if loss_value > accuracy:
        warnings.warn(
            "The searching for the parameters is ended, but the error is "
            f"{float(loss_value):.3e}, larger than the given accuracy {accuracy}. "
            "Consider increase max_step and",
            stacklevel=2,
        )

    return {
        "atomic_smearing": float(smearing),
        "mesh_spacing": float(smooth_mesh_spacing(mesh_spacing)),
        "interpolation_nodes": int(interpolation_nodes),
    }, float(cutoff)
