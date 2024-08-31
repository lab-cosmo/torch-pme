from typing import List, Optional, Union

import torch

from ..lib import generate_kvectors_for_mesh
from ..lib.mesh_interpolator import MeshInterpolator
from .base import CalculatorBaseTorch, PeriodicBase


class _PMEPotentialImpl(PeriodicBase):
    def __init__(
        self,
        exponent: float,
        atomic_smearing: Union[None, float],
        mesh_spacing: Union[None, float],
        interpolation_order: int,
        subtract_self: bool,
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

        # If interior contributions are to be subtracted, also do so for self term
        if self.subtract_interior:
            subtract_self = True
        self.subtract_self = subtract_self

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: Optional[torch.Tensor],
        neighbor_indices: Optional[torch.Tensor],
        neighbor_shifts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Set the defaut values of convergence parameters The total computational cost =
        # cost of SR part + cost of LR part Bigger smearing increases the cost of the SR
        # part while decreasing the cost of the LR part. Since the latter usually is
        # more expensive, we maximize the value of the smearing by default to minimize
        # the cost of the LR part. The auxilary parameter mesh_spacing then controls the
        # convergence of the SR and LR sums, respectively. The default values are chosen
        # to reach a convergence on the order of 1e-4 to 1e-5 for the test structures.
        cell, neighbor_indices, neighbor_shifts, smearing = self._prepare(
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

        if self.mesh_spacing is None:
            mesh_spacing = smearing / 8.0
        else:
            mesh_spacing = self.mesh_spacing

        # Compute short-range (SR) part using a real space sum
        potential_sr = self._compute_sr(
            positions=positions,
            charges=charges,
            cell=cell,
            smearing=smearing,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

        # Compute long-range (LR) part using a Fourier / reciprocal space sum
        potential_lr = self._compute_lr(
            positions=positions,
            charges=charges,
            cell=cell,
            smearing=smearing,
            lr_wavelength=mesh_spacing,
        )

        # Divide by 2 due to double counting of atom pairs
        return (potential_sr + potential_lr) / 2

    def _compute_lr(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        smearing: float,
        lr_wavelength: float,
        subtract_self: bool = True,
    ) -> torch.Tensor:
        # Step 0 (Preparation): Compute number of times each basis vector of the
        # reciprocal space can be scaled until the cutoff is reached
        k_cutoff = 2 * torch.pi / lr_wavelength
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_approx = k_cutoff * basis_norms / 2 / torch.pi
        ns_actual_approx = 2 * ns_approx + 1  # actual number of mesh points
        ns = 2 ** torch.ceil(torch.log2(ns_actual_approx)).long()  # [nx, ny, nz]

        # Step 1: Smear particles onto mesh
        MI = MeshInterpolator(cell, ns, interpolation_order=self.interpolation_order)
        MI.compute_interpolation_weights(positions)
        rho_mesh = MI.points_to_mesh(particle_weights=charges)

        # Step 2: Perform Fourier space convolution (FSC) to get potential on mesh
        # Step 2.1: Generate k-vectors and evaluate kernel function
        # kvectors = self._generate_kvectors(ns=ns, cell=cell)
        kvectors = generate_kvectors_for_mesh(ns=ns, cell=cell)
        knorm_sq = torch.sum(kvectors**2, dim=3)

        # Step 2.2: Evaluate kernel function (careful, tensor shapes are different from
        # the pure Ewald implementation since we are no longer flattening)
        G = self.potential.potential_fourier_from_k_sq(knorm_sq, smearing)
        fill_value = self.potential.potential_fourier_at_zero(smearing)
        G[0, 0, 0] = torch.full([], fill_value, device=G.device)

        potential_mesh = rho_mesh

        # Step 2.3: Perform actual convolution using FFT
        volume = cell.det()
        dims = (1, 2, 3)  # dimensions along which to Fourier transform
        mesh_hat = torch.fft.rfftn(rho_mesh, norm="backward", dim=dims)
        potential_hat = mesh_hat * G
        potential_mesh = torch.fft.irfftn(potential_hat, norm="forward", dim=dims)
        potential_mesh /= volume

        # Step 3: Back interpolation
        interpolated_potential = MI.mesh_to_points(potential_mesh)

        # Step 4: Remove self-contribution if desired
        if subtract_self:
            fill_value = 2.0 / (torch.pi * smearing**2)
            self_contrib = torch.sqrt(torch.full([], fill_value, device=cell.device))
            interpolated_potential -= charges * self_contrib

        return interpolated_potential


class PMEPotential(CalculatorBaseTorch, _PMEPotentialImpl):
    r"""Specie-wise long-range potential using a particle mesh-based Ewald (PME).

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
    :param subtract_self: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from that atom itself (but not
        the periodic images).
    :param subtract_interior: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from all atoms within the cutoff
        Note that if set to true, the self contribution (see previous) is also
        subtracted by default.

    Example
    -------
    We calculate the Madelung constant of a CsCl (Cesium-Chloride) crystal. The
    reference value is :math:`2 \cdot 1.7626 / \sqrt{3} \approx 2.0354`.

    >>> import torch
    >>> from vesin import NeighborList

    Define crystal structure

    >>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    >>> charges = torch.tensor([1.0, -1.0]).reshape(-1, 1)
    >>> cell = torch.eye(3)

    Compute the neighbor indices (``"i"``, ``"j"``) and the neighbor shifts ("``S``")
    using the ``vesin`` package. Refer to the `documentation
    <https://luthaf.fr/vesin>`_ for details on the API. Similarly you can also use
    ``ase``'s :py:func:`neighbor_list <ase.neighborlist.neighbor_list>`.

    >>> cell_dimensions = torch.linalg.norm(cell, dim=1)
    >>> cutoff = torch.min(cell_dimensions) / 2 - 1e-6
    >>> nl = NeighborList(cutoff=cutoff, full_list=True)
    >>> i, j, S = nl.compute(
    ...     points=positions, box=cell, periodic=True, quantities="ijS"
    ... )

    The ``vesin`` calculator returned the indices and the neighbor shifts. We know stack
    the together and convert them into the suitable types

    >>> i = torch.from_numpy(i.astype(int))
    >>> j = torch.from_numpy(j.astype(int))
    >>> neighbor_indices = torch.vstack([i, j])
    >>> neighbor_shifts = torch.from_numpy(S.astype(int))

    If you inspect the neighborlist you will notice that they are empty for the given
    system, which means the the whole potential will be calculated using the long range
    part of the potential. Finally, we initlize the potential class and ``compute`` the
    potential for the crystal

    >>> pme = PMEPotential()
    >>> pme.compute(
    ...     positions=positions,
    ...     charges=charges,
    ...     cell=cell,
    ...     neighbor_indices=neighbor_indices,
    ...     neighbor_shifts=neighbor_shifts,
    ... )
    tensor([[-1.0192],
            [ 1.0192]])

    Which is the close the reference value given above.
    """

    def __init__(
        self,
        exponent: float = 1.0,
        atomic_smearing: Optional[float] = None,
        mesh_spacing: Optional[float] = None,
        interpolation_order: int = 3,
        subtract_self: bool = True,
        subtract_interior: bool = False,
    ):
        _PMEPotentialImpl.__init__(
            self,
            exponent=exponent,
            atomic_smearing=atomic_smearing,
            mesh_spacing=mesh_spacing,
            interpolation_order=interpolation_order,
            subtract_self=subtract_self,
            subtract_interior=subtract_interior,
        )
        CalculatorBaseTorch.__init__(self)

    def compute(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]],
        neighbor_indices: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]],
        neighbor_shifts: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]],
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
        :param neighbor_indices: Optional single or list of 2D tensors of shape ``(2,
            n)``, where ``n`` is the number of atoms. The two rows correspond to the
            indices of a **full neighbor list** for the two atoms which are considered
            neighbors (e.g. within a cutoff distance).
        :param neighbor_shifts: Optional single or list of 2D tensors of shape (3, n),
             where n is the number of atoms. The 3 rows correspond to the shift indices
             for periodic images of a **full neighbor list**.
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
            neighbor_shifts=neighbor_shifts,
        )

    def forward(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]],
        neighbor_indices: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]],
        neighbor_shifts: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )
