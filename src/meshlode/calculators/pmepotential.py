from typing import List, Optional, Union

import torch

from ..lib import generate_kvectors_for_mesh
from ..lib.mesh_interpolator import MeshInterpolator
from .base import CalculatorBaseTorch


class _PMEPotentialImpl:
    def __init__(
        self,
        exponent: float,
        sr_cutoff: Union[None, torch.Tensor],
        atomic_smearing: Union[None, float],
        mesh_spacing: Union[None, float],
        interpolation_order: Union[None, int],
        subtract_self: Union[None, bool],
        subtract_interior: Union[None, bool],
    ):
        # Check that all provided values are correct
        if exponent < 0.0 or exponent > 3.0:
            raise ValueError(f"`exponent` p={exponent} has to satisfy 0 < p < 3")
        if interpolation_order not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_order` from 1 to 5 are allowed")
        if atomic_smearing is not None and atomic_smearing <= 0:
            raise ValueError(f"`atomic_smearing` {atomic_smearing} has to be positive")

        self.atomic_smearing = atomic_smearing
        self.mesh_spacing = mesh_spacing
        self.interpolation_order = interpolation_order
        self.sr_cutoff = sr_cutoff

        # If interior contributions are to be subtracted, also do so for self term
        if subtract_interior:
            subtract_self = True
        self.subtract_self = subtract_self
        self.subtract_interior = subtract_interior

        self.atomic_smearing = atomic_smearing
        self.mesh_spacing = mesh_spacing
        self.interpolation_order = interpolation_order
        self.sr_cutoff = sr_cutoff

        # If interior contributions are to be subtracted, also do so for self term
        if subtract_interior:
            subtract_self = True
        self.subtract_self = subtract_self
        self.subtract_interior = subtract_interior

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        neighbor_indices: Optional[torch.Tensor] = None,
        neighbor_shifts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the "electrostatic" potential at the position of all atoms in a
        structure.

        :param positions: torch.tensor of shape (n_atoms, 3). Contains the Cartesian
            coordinates of the atoms. The implementation also works if the positions
            are not contained within the unit cell.
        :param charges: torch.tensor of shape `(n_atoms, n_channels)`. In the simplest
            case, this would be a tensor of shape (n_atoms, 1) where charges[i,0] is the
            charge of atom i. More generally, the potential for the same atom positions
            is computed for n_channels independent meshes, and one can specify the
            "charge" of each atom on each of the meshes independently. For standard LODE
            that treats all (atomic) types separately, one example could be: If n_atoms
            = 4 and the types are [Na, Cl, Cl, Na], one could set n_channels=2 and use
            the one-hot encoding charges = torch.tensor([[1,0],[0,1],[0,1],[1,0]]) for
            the charges. This would then separately compute the "Na" potential and "Cl"
            potential. Subtracting these from each other, one could recover the more
            standard electrostatic potential in which Na and Cl have charges of +1 and
            -1, respectively.
        :param cell: torch.tensor of shape `(3, 3)`. Describes the unit cell of the
            structure, where cell[i] is the i-th basis vector.
        :param neighbor_indices: Optional single or list of 2D tensors of shape (2, n),
            where n is the number of atoms. The 2 rows correspond to the indices of
            the two atoms which are considered neighbors (e.g. within a cutoff distance)
        :param neighbor_shifts: Optional single or list of 2D tensors of shape (3, n),
             where n is the number of atoms. The 3 rows correspond to the shift indices
             for periodic images.

        :returns: torch.tensor of shape `(n_atoms, n_channels)` containing the potential
        at the position of each atom for the `n_channels` independent meshes separately.
        """
        # Set the defaut values of convergence parameters
        # The total computational cost = cost of SR part + cost of LR part
        # Bigger smearing increases the cost of the SR part while decreasing the cost
        # of the LR part. Since the latter usually is more expensive, we maximize the
        # value of the smearing by default to minimize the cost of the LR part.
        # The two auxilary parameters (sr_cutoff, lr_wavelength) then control the
        # convergence of the SR and LR sums, respectively. The default values are
        # chosen to reach a convergence on the order of 1e-4 to 1e-5 for the test
        # structures.
        if self.sr_cutoff is None:
            cell_dimensions = torch.linalg.norm(cell, dim=1)
            cutoff_max = torch.min(cell_dimensions) / 2 - 1e-6
            sr_cutoff = cutoff_max
        else:
            sr_cutoff = self.sr_cutoff

        if self.atomic_smearing is None:
            smearing = sr_cutoff / 5.0
        else:
            smearing = self.atomic_smearing

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
            sr_cutoff=sr_cutoff,
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

        # Combine both parts to obtain the full potential
        potential_ewald = potential_sr + potential_lr
        return potential_ewald

    def _compute_lr(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        smearing: torch.Tensor,
        lr_wavelength: torch.Tensor,
        subtract_self=True,
    ) -> torch.Tensor:
        """
        Compute the long-range part of the Ewald sum in realspace

        :param positions: torch.tensor of shape (n_atoms, 3). Contains the Cartesian
            coordinates of the atoms. The implementation also works if the positions
            are not contained within the unit cell.
        :param charges: torch.tensor of shape `(n_atoms, n_channels)`. In the simplest
            case, this would be a tensor of shape (n_atoms, 1) where charges[i,0] is the
            charge of atom i. More generally, the potential for the same atom positions
            is computed for n_channels independent meshes, and one can specify the
            "charge" of each atom on each of the meshes independently.
        :param cell: torch.tensor of shape `(3, 3)`. Describes the unit cell of the
            structure, where cell[i] is the i-th basis vector.
        :param smearing: torch.Tensor smearing paramter determining the splitting
            between the SR and LR parts.
        :param lr_wavelength: Spatial resolution used for the long-range (reciprocal
            space) part of the Ewald sum. More conretely, all Fourier space vectors with
            a wavelength >= this value will be kept.

        :returns: torch.tensor of shape `(n_atoms, n_channels)` containing the potential
        at the position of each atom for the `n_channels` independent meshes separately.
        """
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
        G[0, 0, 0] = self.potential.potential_fourier_at_zero(smearing)

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
            self_contrib = (
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=cell.device)) / smearing
            )
            interpolated_potential -= charges * self_contrib

        return interpolated_potential


class PMEPotential(CalculatorBaseTorch, _PMEPotentialImpl):
    r"""Specie-wise long-range potential using a particle mesh-based Ewald (PME).

    Scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N` used as a reference to test faster implementations.

    :param all_types: Optional global list of all atomic types that should be considered
        for the computation. This option might be useful when running the calculation on
        subset of a whole dataset and it required to keep the shape of the output
        consistent. If this is not set the possible atomic types will be determined when
        calling the :meth:`compute()`.
    :param exponent: the exponent "p" in 1/r^p potentials
    :param sr_cutoff: Cutoff radius used for the short-range part of the Ewald sum. If
        not set to a global value, it will be set to be half of the shortest lattice
        vector defining the cell (separately for each structure).
    :param atomic_smearing: Width of the atom-centered Gaussian used to split the
        Coulomb potential into the short- and long-range parts. If not set to a global
        value, it will be set to 1/5 times the sr_cutoff value (separately for each
        structure) to ensure convergence of the short-range part to a relative precision
        of 1e-5.
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
    """

    def __init__(
        self,
        all_types: Optional[List[int]] = None,
        exponent: float = 1.0,
        sr_cutoff: Optional[torch.Tensor] = None,
        atomic_smearing: Optional[float] = None,
        mesh_spacing: Optional[float] = None,
        interpolation_order: Optional[int] = 3,
        subtract_self: Optional[bool] = True,
        subtract_interior: Optional[bool] = False,
    ):
        _PMEPotentialImpl.__init__(
            self,
            exponent=exponent,
            sr_cutoff=sr_cutoff,
            atomic_smearing=atomic_smearing,
            mesh_spacing=mesh_spacing,
            interpolation_order=interpolation_order,
            subtract_self=subtract_self,
            subtract_interior=subtract_interior,
        )
        CalculatorBaseTorch.__init__(self, all_types=all_types, exponent=exponent)

    def compute(
        self,
        types: Union[List[torch.Tensor], torch.Tensor],
        positions: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
        charges: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor] = None,
        neighbor_shifts: Union[List[torch.Tensor], torch.Tensor] = None,
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
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

    def forward(
        self,
        types: Union[List[torch.Tensor], torch.Tensor],
        positions: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
        charges: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor] = None,
        neighbor_shifts: Union[List[torch.Tensor], torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(
            types=types,
            positions=positions,
            cell=cell,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )
