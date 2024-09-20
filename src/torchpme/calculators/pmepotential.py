from typing import List, Optional, Union

import torch

from ..lib import generate_kvectors_for_mesh
from ..lib.mesh_interpolator import MeshInterpolator
from ..lib.potentials import gamma
from .base import CalculatorBaseTorch, PeriodicBase


class _PMEPotentialImpl(PeriodicBase):
    def __init__(
        self,
        exponent: float,
        atomic_smearing: Optional[float],
        mesh_spacing: Optional[float],
        interpolation_order: int,
        subtract_interior: bool,
    ):
        if interpolation_order not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_order` from 1 to 5 are allowed")

        PeriodicBase.__init__(
            self,
            exponent=exponent,
            atomic_smearing=atomic_smearing,
            subtract_interior=subtract_interior,
        )
        self.mesh_spacing = mesh_spacing
        self.interpolation_order = interpolation_order

        # TorchScript requires to initialize all attributes in __init__
        self._cell_cache = -1 * torch.ones([3, 3])
        self._MI = MeshInterpolator(
            cell=torch.eye(3),
            ns_mesh=torch.ones(3),
            interpolation_order=self.interpolation_order,
        )
        self._G = torch.zeros(1)

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
        smearing = self._estimate_smearing(cell)

        if self.mesh_spacing is None:
            mesh_spacing = smearing / 8.0
        else:
            mesh_spacing = self.mesh_spacing

        # Compute short-range (SR) part using a real space sum
        potential_sr = self._compute_sr(
            smearing=smearing,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
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

        dtype = positions.dtype
        device = positions.device
        self._cell_cache = self._cell_cache.to(dtype=dtype, device=device)

        # Inverse of cell volume for later use
        ivolume = torch.abs(cell.det()).pow(-1)

        # Kernel function `G` and initialization of `MeshInterpolator` only depend on
        # `cell`. Caching can save up to 15%.
        if not torch.allclose(cell, self._cell_cache):
            self._cell_cache = cell.detach().clone()

            # Init 0 (Preparation): Compute number of times each basis vector of the
            # reciprocal space can be scaled until the cutoff is reached
            k_cutoff = 2 * torch.pi / lr_wavelength
            basis_norms = torch.linalg.norm(cell, dim=1)
            ns_approx = k_cutoff * basis_norms / 2 / torch.pi
            ns_actual_approx = 2 * ns_approx + 1  # actual number of mesh points
            # ns = [nx, ny, nz]
            ns = torch.tensor(2).pow(torch.ceil(torch.log2(ns_actual_approx)).long())

            # Init 1: Initialize mesh interpolator
            self._MI = MeshInterpolator(
                cell, ns, interpolation_order=self.interpolation_order
            )

            # Init 2: Perform Fourier space convolution (FSC) to get kernel on mesh
            # 2.1: Generate k-vectors and evaluate kernel function
            kvectors = generate_kvectors_for_mesh(ns=ns, cell=cell)
            knorm_sq = torch.sum(kvectors**2, dim=3)

            # 2.2: Evaluate kernel function (careful, tensor shapes are different from
            # the pure Ewald implementation since we are no longer flattening)
            # pre-scale with volume to save some multiplications further down
            self._G = (
                self.potential.potential_fourier_from_k_sq(knorm_sq, smearing) * ivolume
            )
            fill_value = self.potential.potential_fourier_at_zero(smearing)
            self._G[0, 0, 0] = torch.full([], fill_value, device=device)

        # Step 1. Compute density interpolation
        self._MI.compute_interpolation_weights(positions)
        rho_mesh = self._MI.points_to_mesh(particle_weights=charges)

        # Step 2: Perform actual convolution using FFT
        dims = (1, 2, 3)  # dimensions along which to Fourier transform
        # convolution with the kernel function `G`
        rho_hat = self._G * torch.fft.rfftn(rho_mesh, norm="backward", dim=dims)
        potential_mesh = torch.fft.irfftn(rho_hat, norm="forward", dim=dims)

        # Step 3: Back interpolation
        interpolated_potential = self._MI.mesh_to_points(potential_mesh)

        # Step 4: Remove the self-contribution: Using the Coulomb potential as an
        # example, this is the potential generated at the origin by the fictituous
        # Gaussian charge density in order to split the potential into a SR and LR part.
        # This contribution always should be subtracted since it depends on the smearing
        # parameter, which is purely a convergence parameter.
        phalf = self.exponent / 2
        fill_value = 1 / gamma(torch.tensor(phalf + 1)) / (2 * smearing**2) ** phalf
        self_contrib = torch.full([], fill_value, device=device)
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


class PMEPotential(CalculatorBaseTorch, _PMEPotentialImpl):
    r"""Potential using a particle mesh-based Ewald (PME).

    Scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N` used as a reference to test faster implementations.

    For computing a **neighborlist** a reasonable ``cutoff`` is half the length of
    the shortest cell vector, which can be for example computed according as

    .. code-block:: python

        cell_dimensions = torch.linalg.norm(cell, dim=1)
        cutoff = torch.min(cell_dimensions) / 2 - 1e-6

    This ensures a accuracy of the short range part of ``1e-5``.

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials
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

    >>> pme = PMEPotential()
    >>> pme.compute(
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
        exponent: float = 1.0,
        atomic_smearing: Optional[float] = None,
        mesh_spacing: Optional[float] = None,
        interpolation_order: int = 3,
        subtract_interior: bool = False,
    ):
        _PMEPotentialImpl.__init__(
            self,
            exponent=exponent,
            atomic_smearing=atomic_smearing,
            mesh_spacing=mesh_spacing,
            interpolation_order=interpolation_order,
            subtract_interior=subtract_interior,
        )
        CalculatorBaseTorch.__init__(self)

    def compute(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor],
        neighbor_distances: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute potential for all provided "systems" stacked inside list.

        The computation is performed on the same ``device`` as ``dtype`` is the input is
        stored on. The ``dtype`` of the output tensors will be the same as the input.

        :param positions: Single or 2D tensor of shape (``len(charges), 3``) containing
            the Cartesian positions of all point charges in the system.
        :param charges: Single 2D tensor or list of 2D tensor of shape (``n_channels,
            len(positions))``. ``n_channels`` is the number of charge channels the
            potential should be calculated for a standard potential ``n_channels=1``. If
            more than one "channel" is provided multiple potentials for the same
            position but different are computed.
        :param cell: single or 2D tensor of shape (3, 3), describing the bounding
            box/unit cell of the system. Each row should be one of the bounding box
            vector; and columns should contain the x, y, and z components of these
            vectors (i.e. the cell should be given in row-major order).
        :param neighbor_indices: Single or list of 2D tensors of shape ``(n, 2)``, where
            ``n`` is the number of neighbors. The two columns correspond to the indices
            of a **half neighbor list** for the two atoms which are considered neighbors
            (e.g. within a cutoff distance).
        :param neighbor_distances: single or list of 1D tensors containing the distance
            between the ``n`` pairs corresponding to a **half neighbor list**.
        :return: Single or list of torch tensors containing the potential(s) for all
            positions. Each tensor in the list is of shape ``(len(positions),
            len(charges))``, where If the inputs are only single tensors only a single
            torch tensor with the potentials is returned.
        """

        return self._compute_impl(
            positions=positions,
            cell=cell,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )

    def forward(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor],
        neighbor_distances: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )
