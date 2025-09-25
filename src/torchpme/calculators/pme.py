from typing import Optional

import torch

from ..lib.kspace_filter import KSpaceFilter
from ..lib.kvectors import get_ns_mesh
from ..lib.mesh_interpolator import MeshInterpolator
from ..potentials import Potential
from .calculator import Calculator


class PMECalculator(Calculator):
    r"""
    Potential using a particle mesh-based Ewald (PME).

    Scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N` used as a reference to test faster implementations.

    For getting reasonable values for the ``smaring`` of the potential class and  the
    ``mesh_spacing`` based on a given accuracy for a specific structure you should use
    :func:`torchpme.tuning.tune_pme`. This function will also find the optimal
    ``cutoff`` for the  **neighborlist**.

    .. hint::

        For a training exercise it is recommended only run a tuning procedure with
        :func:`torchpme.tuning.tune_pme` for the largest system in your dataset.

    :param potential: A :class:`torchpme.potentials.Potential` object that implements
        the evaluation of short and long-range potential terms. The ``smearing``
        parameter of the potential determines the split between real and k-space
        regions. For a :class:`torchpme.CoulombPotential` it corresponds to the
        smearing of the atom-centered Gaussian used to split the Coulomb potential into
        the short- and long-range parts. A reasonable value for most systems is to set
        it to ``1/5`` times the neighbor list cutoff.
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically be set
        to half of ``smearing``.
    :param interpolation_nodes: The number ``n`` of nodes used in the interpolation per
        coordinate axis. The total number of interpolation nodes in 3D will be ``n^3``.
        In general, for ``n`` nodes, the interpolation will be performed by piecewise
        polynomials of degree ``n - 1`` (e.g. ``n = 4`` for cubic interpolation).
        Only the values ``3, 4, 5, 6, 7`` are supported.
    :param full_neighbor_list: If set to :obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :obj:`False`, a "half" neighbor list is expected.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
    """

    def __init__(
        self,
        potential: Potential,
        mesh_spacing: float,
        interpolation_nodes: int = 4,
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
    ):
        super().__init__(
            potential=potential,
            full_neighbor_list=full_neighbor_list,
            prefactor=prefactor,
        )

        if potential.smearing is None:
            raise ValueError(
                "Must specify smearing to use a potential with PMECalculator"
            )

        self.mesh_spacing: float = mesh_spacing

        cell = torch.eye(
            3,
            device=self.potential.smearing.device,
            dtype=self.potential.smearing.dtype,
        )
        ns_mesh = torch.ones(3, dtype=int, device=cell.device)

        self.kspace_filter: KSpaceFilter = KSpaceFilter(
            cell=cell,
            ns_mesh=ns_mesh,
            kernel=self.potential,
            fft_norm="backward",
            ifft_norm="forward",
        )

        self.interpolation_nodes: int = interpolation_nodes

        self.mesh_interpolator: MeshInterpolator = MeshInterpolator(
            cell=cell,
            ns_mesh=ns_mesh,
            interpolation_nodes=self.interpolation_nodes,
            method="Lagrange",  # convention for classic PME
        )

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        periodic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if periodic is None:
            periodic = torch.tensor([True, True, True], device=cell.device)

        n_periodic = torch.sum(periodic).item()
        if n_periodic == 3:
            periodicity = 3
            nonperiodic_axis = None
        elif n_periodic == 2:
            periodicity = 2
            nonperiodic_axis = torch.where(~periodic)[0]
            max_distance = torch.max(positions[:, nonperiodic_axis]) - torch.min(
                positions[:, nonperiodic_axis]
            )
            cell_size = torch.linalg.norm(cell[nonperiodic_axis])
            if max_distance > cell_size / 3:
                raise ValueError(
                    f"Maximum distance along non-periodic axis ({max_distance}) "
                    f"exceeds one third of cell size ({cell_size})."
                )
        else:
            raise ValueError(
                "K-space summation is not implemented for 1D or non-periodic systems."
            )

        # TODO: Kernel function `G` and initialization of `MeshInterpolator` only depend
        # on `cell`. Caching may save up to 15% but issues with AD need to be resolved.

        # Compute number of times each basis vector of the reciprocal space can be
        # scaled until the cutoff is reached
        ns = get_ns_mesh(cell, self.mesh_spacing)

        self.mesh_interpolator.update(cell, ns)

        self.kspace_filter.update(cell, ns)

        self.mesh_interpolator.compute_weights(positions)
        rho_mesh = self.mesh_interpolator.points_to_mesh(particle_weights=charges)

        potential_mesh = self.kspace_filter.forward(rho_mesh)

        ivolume = torch.abs(cell.det()).pow(-1)
        interpolated_potential = (
            self.mesh_interpolator.mesh_to_points(potential_mesh) * ivolume
        )

        # Using the Coulomb potential as an example, this is the potential generated
        # at the origin by the fictituous Gaussian charge density in order to split
        # the potential into a SR and LR part. This contribution always should be
        # subtracted since it depends on the smearing parameter, which is purely a
        # convergence parameter.
        interpolated_potential -= charges * self.potential.self_contribution()

        # If the cell has a net charge (i.e. if sum(charges) != 0), the method
        # implicitly assumes that a homogeneous background charge of the opposite
        # sign is present to make the cell neutral. In this case, the potential has
        # to be adjusted to compensate for this. An extra factor of 2 is added to
        # compensate for the division by 2 later on
        charge_tot = torch.sum(charges, dim=0)
        prefac = self.potential.background_correction()
        interpolated_potential -= 2 * prefac * charge_tot * ivolume

        if periodicity == 2:
            axis = nonperiodic_axis
            z_i = positions[:, axis].view(-1, 1)
            basis_len = torch.linalg.norm(cell[axis])
            M_axis = torch.sum(charges * z_i, dim=0)
            M_axis_sq = torch.sum(charges * z_i**2, dim=0)
            V = torch.abs(torch.linalg.det(cell))
            E_slab = (4.0 * torch.pi / V) * (
                z_i * M_axis
                - 0.5 * (M_axis_sq + charge_tot * z_i**2)
                - charge_tot / 12.0 * basis_len**2
            )
            interpolated_potential += E_slab

        # Compensate for double counting of pairs (i,j) and (j,i)
        return interpolated_potential / 2
