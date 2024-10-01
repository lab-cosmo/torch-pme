from typing import Optional, Union

import torch

from ..lib.kspace_filter import KSpaceFilter
from ..lib.kvectors import get_ns_mesh
from ..lib.mesh_interpolator import MeshInterpolator
from ..lib.potentials import gamma
from .base import CalculatorBaseTorch


class PMEPotential(CalculatorBaseTorch):
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

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param atomic_smearing: Width of the atom-centered Gaussian used to split the
        Coulomb potential into the short- and long-range parts. A reasonable value for
        most systems is to set it to ``1/5`` times the neighbor list cutoff. If
        :py:obj:`None` ,it will be set to 1/5 times of half the largest box vector
        (separately for each structure).
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically be set
        to half of ``atomic_smearing``.
    :param interpolation_order: Interpolation order for mapping onto the grid, where an
        interpolation order of p corresponds to interpolation by a polynomial of degree
        ``p - 1`` (e.g. ``p = 4`` for cubic interpolation).
    :param subtract_interior: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from all atoms within the cutoff
        Note that if set to true, the self contribution (see previous) is also
        subtracted by default.

    Example
    -------
    We calculate the Madelung constant of a CsCl (Cesium-Chloride) crystal. The
    reference value is :math:`2 \cdot 1.7626 / \sqrt{3} \approx 2.0354`.

    >>> import torch
    >>> from torchpme.lib.mesh_interpolator import MeshInterpolator
    >>> from vesin.torch import NeighborList

    Define crystal structure

    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)

    Compute the neighbor indices (``"i"``, ``"j"``) and the neighbor distances ("``d``")
    using the ``vesin`` package. Refer to the `documentation <https://luthaf.fr/vesin>`_
    for details on the API.

    >>> cell_dimensions = torch.linalg.norm(cell, dim=1)
    >>> cutoff = torch.min(cell_dimensions) / 2 - 1e-6
    >>> nl = NeighborList(cutoff=cutoff, full_list=False)
    >>> i, j, neighbor_distances = nl.compute(
    ...     points=positions, box=cell, periodic=True, quantities="ijd"
    ... )
    >>> neighbor_indices = torch.stack([i, j], dim=1)

    If you inspect the neighbor list you will notice that the tensors are empty for the
    given system, which means the the whole potential will be calculated using the long
    range part of the potential. Finally, we initlize the potential class and
    ``compute`` the potential for the crystal.

    >>> interpolator = MeshInterpolator().to(cell.device, cell.dtype)
    >>> pme = PMEPotential(interpolator)
    >>> pme.forward(
    ...     positions=positions,
    ...     charges=charges,
    ...     cell=cell,
    ...     neighbor_indices=neighbor_indices,
    ...     neighbor_distances=neighbor_distances,
    ... )
    tensor([[-1.0192],
            [ 1.0192]], dtype=torch.float64)

    Which is close to the reference value given above.

    """

    def __init__(
        self,
        interpolator: MeshInterpolator,
        exponent: float = 1.0,
        atomic_smearing: Union[float, torch.Tensor, None] = None,
        mesh_spacing: Optional[float] = None,
        interpolation_order: int = 3,
        subtract_interior: bool = False,
        full_neighbor_list: bool = False,
    ):
        super().__init__(
            exponent=exponent,
            smearing=atomic_smearing,
            full_neighbor_list=full_neighbor_list,
        )

        self.interpolator = interpolator
        self.mesh_spacing = mesh_spacing
        self.subtract_interior = subtract_interior

        if atomic_smearing is not None and atomic_smearing <= 0:
            raise ValueError(f"`atomic_smearing` {atomic_smearing} has to be positive")
        self.atomic_smearing = atomic_smearing

        if interpolation_order not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_order` from 1 to 5 are allowed")
        self.interpolation_order = interpolation_order

        # Initialize the filter module. Set dummy value for smearing to propper
        # initilize the `KSpaceFilter` below
        if self.atomic_smearing is None:
            self.potential.smearing = 1.0
        self._KF = KSpaceFilter(
            cell=torch.eye(3),
            ns_mesh=torch.ones(3, dtype=int),
            kernel=self.potential,
            fft_norm="backward",
            ifft_norm="forward",
        )

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        # Set the defaut values of convergence parameters The total computational cost =
        # cost of SR part + cost of LR part Bigger smearing increases the cost of the SR
        # part while decreasing the cost of the LR part. Since the latter usually is
        # more expensive, we maximize the value of the smearing by default to minimize
        # the cost of the LR part. The auxilary parameter mesh_spacing then controls the
        # convergence of the SR and LR sums, respectively. The default values are chosen
        # to reach a convergence on the order of 1e-4 to 1e-5 for the test structures.
        if self.atomic_smearing is None:
            smearing = self.estimate_smearing(cell)
        else:
            smearing = self.atomic_smearing

        if self.mesh_spacing is None:
            mesh_spacing = smearing / 8.0
        else:
            mesh_spacing = self.mesh_spacing

        # Compute short-range (SR) part using a real space sum
        potential_sr = self._compute_sr(
            is_periodic=True,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            subtract_interior=self.subtract_interior,
        )

        # Compute long-range (LR) part using a Fourier / reciprocal space sum
        potential_lr = self._compute_lr(
            positions=positions,
            charges=charges,
            cell=cell,
            smearing=smearing,
            lr_wavelength=mesh_spacing,
        )

        return potential_sr + potential_lr

    def _compute_lr(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        smearing: float,
        lr_wavelength: float,
    ) -> torch.Tensor:
        # TODO: Kernel function `G` and initialization of `MeshInterpolator` only depend on
        # `cell`. Caching may save up to 15% but issues with AD should be taken
        # resolved.

        # Init 0 (Preparation): Compute number of times each basis vector of the
        # reciprocal space can be scaled until the cutoff is reached
        ns = get_ns_mesh(cell, lr_wavelength)

        # Update the mesh for the k-space filter and the interpolator
        self.potential.smearing = smearing
        self._KF.update_mesh(cell, ns)
        self.interpolator.update_mesh(cell, ns)

        # Step 1. Compute density interpolation
        self.interpolator.compute_weights(positions)
        rho_mesh = self.interpolator.points_to_mesh(particle_weights=charges)

        # Step 2: Perform actual convolution using FFT
        potential_mesh = self._KF.compute(rho_mesh)

        # Step 3: Back interpolation, and apply cell volume scaling
        ivolume = torch.abs(cell.det()).pow(-1)
        interpolated_potential = self.interpolator.mesh_to_points(potential_mesh) * ivolume

        # Step 4: Remove the self-contribution: Using the Coulomb potential as an
        # example, this is the potential generated at the origin by the fictituous
        # Gaussian charge density in order to split the potential into a SR and LR part.
        # This contribution always should be subtracted since it depends on the smearing
        # parameter, which is purely a convergence parameter.
        phalf = self.exponent / 2
        fill_value = 1 / gamma(torch.tensor(phalf + 1)) / (2 * smearing**2) ** phalf
        self_contrib = torch.full([], fill_value, device=self._device)
        interpolated_potential -= charges * self_contrib

        # Step 5: The method requires that the unit cell is charge-neutral.
        # If the cell has a net charge (i.e. if sum(charges) != 0), the method
        # implicitly assumes that a homogeneous background charge of the opposite sign
        # is present to make the cell neutral. In this case, the potential has to be
        # adjusted to compensate for this.
        # An extra factor of 2 is added to compensate for the division by 2 later on
        charge_tot = torch.sum(charges, dim=0)
        prefac = torch.pi**1.5 * (2 * smearing**2) ** ((3 - self.exponent) / 2)
        prefac /= (3 - self.exponent) * gamma(torch.tensor(self.exponent / 2))
        interpolated_potential -= 2 * prefac * charge_tot * ivolume

        # Compensate for double counting of pairs (i,j) and (j,i)
        return interpolated_potential / 2
