from typing import List, Optional, Union

import torch

from ..lib.fourier_convolution import FourierSpaceConvolution
from ..lib.mesh_interpolator import MeshInterpolator
from .base import CalculatorBase


class MeshPotential(CalculatorBase):
    r"""Specie-wise long-range potential, computed on a grid.

    Method scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N`. This class does not perform a usual Ewald style splitting into a short
    and long range contribution but calculates the full contribution to the potential on
    a grid.

    For a Particle Mesh Ewald (PME) use :py:class:`meshlode.PMEPotential`.

    :param atomic_smearing: Width of the atom-centered Gaussian used to create the
        atomic density.
    :param all_types: Optional global list of all atomic types that should be considered
        for the computation. This option might be useful when running the calculation on
        subset of a whole dataset and it required to keep the shape of the output
        consistent. If this is not set the possible atomic types will be determined when
        calling the :meth:`compute()`.
    :param exponent: the exponent "p" in 1/r^p potentials
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically be set
        to half of ``atomic_smearing``.
    :param interpolation_order: Interpolation order for mapping onto the grid, where an
        interpolation order of p corresponds to interpolation by a polynomial of degree
        ``p - 1`` (e.g. ``p = 4`` for cubic interpolation).
    :param subtract_self: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from that atom itself (but not
        the periodic images).

    Example
    -------
    >>> import torch
    >>> from meshlode import MeshPotential

    Define simple example structure having the CsCl (Cesium Chloride) structure

    >>> types = torch.tensor([55, 17])  # Cs and Cl
    >>> positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> cell = torch.eye(3)

    Compute features

    >>> MP = MeshPotential(atomic_smearing=0.2, mesh_spacing=0.1, interpolation_order=4)
    >>> MP.compute(types=types, positions=positions, cell=cell)
    tensor([[-0.5467,  1.3755],
            [ 1.3755, -0.5467]])
    """

    def __init__(
        self,
        atomic_smearing: float,
        all_types: Optional[List[int]] = None,
        exponent: float = 1.0,
        mesh_spacing: Optional[float] = None,
        interpolation_order: Optional[int] = 4,
        subtract_self: Optional[bool] = False,
    ):
        super().__init__(all_types=all_types, exponent=exponent)

        # Check that all provided values are correct
        if interpolation_order not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_order` from 1 to 5 are allowed")

        # If no explicit mesh_spacing is given, set it such that it can resolve
        # the smeared potentials.
        if atomic_smearing <= 0:
            raise ValueError(f"`atomic_smearing` {atomic_smearing} has to be positive")

        self.atomic_smearing = atomic_smearing
        self.mesh_spacing = mesh_spacing
        self.interpolation_order = interpolation_order
        self.subtract_self = subtract_self

        # Initilize auxiliary objects
        self.fourier_space_convolution = FourierSpaceConvolution()

    def compute(
        self,
        types: Union[List[torch.Tensor], torch.Tensor],
        positions: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
        charges: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute potential for all provided "systems" stacked inside list.

        The computation is performed on the same ``device`` as ``systems`` is stored on.
        The ``dtype`` of the output tensors will be the same as the input.

        :param types: single or list of 1D tensor of integer representing the
            particles identity. For atoms, this is typically their atomic numbers.
        :param positions: single or 2D tensor of shape (len(types), 3) containing the
            Cartesian positions of all particles in the system.
        :param cell: single or 2D tensor of shape (3, 3), describing the bounding
            box/unit cell of the system. Each row should be one of the bounding box
            vector; and columns should contain the x, y, and z components of these
            vectors (i.e. the cell should be given in row-major order).
        :param charges: Optional single or list of 2D tensor of shape (len(types), n),
        :param neighbor_indices: Optional single or list of 2D tensors of shape (2, n),
            where n is the number of atoms. The 2 rows correspond to the indices of
            the two atoms which are considered neighbors (e.g. within a cutoff distance)
        :param neighbor_shifts: Optional single or list of 2D tensors of shape (3, n),
             where n is the number of atoms. The 3 rows correspond to the shift indices
             for periodic images.

        :return: List of torch Tensors containing the potentials for all frames and all
            atoms. Each tensor in the list is of shape (n_atoms, n_types), where
            n_types is the number of types in all systems combined. If the input was
            a single system only a single torch tensor with the potentials is returned.

            IMPORTANT: If multiple types are present, the different "types-channels"
            are ordered according to atomic number. For example, if a structure contains
            a water molecule with atoms 0, 1, 2 being of types O, H, H, then for this
            system, the feature tensor will be of shape (3, 2) = (``n_atoms``,
            ``n_types``), where ``features[0, 0]`` is the potential at the position of
            the Oxygen atom (atom 0, first index) generated by the HYDROGEN(!) atoms,
            while ``features[0,1]`` is the potential at the position of the Oxygen atom
            generated by the Oxygen atom(s).
        """

        return self._compute_impl(
            types=types,
            positions=positions,
            cell=cell,
            charges=charges,
            neighbor_indices=None,
            neighbor_shifts=None,
        )

    # This function is kept to keep MeshLODE compatible with the broader pytorch
    # infrastructure, which require a "forward" function. We name this function
    # "compute" instead, for compatibility with other COSMO software.
    def forward(
        self,
        types: Union[List[torch.Tensor], torch.Tensor],
        positions: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
        charges: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(
            types=types,
            positions=positions,
            cell=cell,
            charges=charges,
        )

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        cell: Union[None, torch.Tensor],
        charges: torch.Tensor,
        neighbor_indices: Union[None, torch.Tensor],
        neighbor_shifts: Union[None, torch.Tensor],
    ) -> torch.Tensor:

        if self.mesh_spacing is None:
            mesh_spacing = self.atomic_smearing / 2
        else:
            mesh_spacing = self.mesh_spacing

        # Initializations
        k_cutoff = 2 * torch.pi / mesh_spacing

        # Compute number of times each basis vector of the
        # reciprocal space can be scaled until the cutoff
        # is reached
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_approx = k_cutoff * basis_norms / 2 / torch.pi
        ns_actual_approx = 2 * ns_approx + 1  # actual number of mesh points
        ns = 2 ** torch.ceil(torch.log2(ns_actual_approx)).long()  # [nx, ny, nz]

        # Step 1: Smear particles onto mesh
        MI = MeshInterpolator(cell, ns, interpolation_order=self.interpolation_order)
        MI.compute_interpolation_weights(positions)
        rho_mesh = MI.points_to_mesh(particle_weights=charges)

        # Step 2: Perform Fourier space convolution (FSC)
        potential_mesh = self.fourier_space_convolution.compute(
            mesh_values=rho_mesh,
            cell=cell,
            potential_exponent=1,
            atomic_smearing=self.atomic_smearing,
        )

        # Step 3: Back interpolation
        interpolated_potential = MI.mesh_to_points(potential_mesh)

        # Remove self contribution
        if self.subtract_self:
            self_contrib = (
                torch.sqrt(
                    torch.tensor(
                        2.0 / torch.pi, dtype=positions.dtype, device=positions.device
                    ),
                )
                / self.atomic_smearing
            )
            interpolated_potential -= charges * self_contrib

        return interpolated_potential
