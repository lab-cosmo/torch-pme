import torch
from torch import profiler

from ..lib import Potential
from ..lib.kspace_filter import KSpaceFilter
from ..lib.kvectors import get_ns_mesh
from ..lib.mesh_interpolator import MeshInterpolator
from .base import Calculator


class PMECalculator(Calculator):
    r"""
    Potential using a particle mesh-based Ewald (PME).

    Scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N` used as a reference to test faster implementations.

    For getting reasonable values for a given accuracy you should use
    :func:`torchpme.utils.tune_pme`, which will also find a reasonable ``cutoff`` for
    the  **neighborlist**.

    For a training exercise it is recommended only run the tuning
    procedure for the largest system in your dataset.

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
        mesh_spacing: float,
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
