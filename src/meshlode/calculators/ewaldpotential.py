from typing import List, Optional, Union

import torch

# extra imports for neighbor list
from ase import Atoms
from ase.neighborlist import neighbor_list

from .base import CalculatorBase


class EwaldPotential(CalculatorBase):
    r"""Specie-wise long-range potential computed using the Ewald sum.

    Scaling as :math:`\mathcal{O}(N^2)` with respect to the number of particles
    :math:`N`.

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
    :param lr_wavelength: Spatial resolution used for the long-range (reciprocal space)
        part of the Ewald sum. More conretely, all Fourier space vectors with a
        wavelength >= this value will be kept. If not set to a global value, it will be
        set to half the atomic_smearing parameter to ensure convergence of the
        long-range part to a relative precision of 1e-5.
    :param subtract_self: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from that atom itself (but not
        the periodic images).
    :param subtract_interior: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from all atoms within the cutoff
        Note that if set to true, the self contribution (see previous) is also
        subtracted by default.

    Example
    -------
    >>> import torch
    >>> from meshlode import EwaldPotential

    Define simple example structure having the CsCl (Cesium Chloride) structure

    >>> types = torch.tensor([55, 17])  # Cs and Cl
    >>> positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> cell = torch.eye(3)

    Compute features

    >>> EP = EwaldPotential()
    >>> EP.compute(types=types, positions=positions, cell=cell)
    tensor([[-0.7391, -2.7745],
            [-2.7745, -0.7391]])
    """

    def __init__(
        self,
        all_types: Optional[List[int]] = None,
        exponent: float = 1.0,
        sr_cutoff: Optional[torch.Tensor] = None,
        atomic_smearing: Optional[float] = None,
        lr_wavelength: Optional[float] = None,
        subtract_self: Optional[bool] = True,
        subtract_interior: Optional[bool] = False,
    ):
        super().__init__(all_types=all_types, exponent=exponent)

        if atomic_smearing is not None and atomic_smearing <= 0:
            raise ValueError(f"`atomic_smearing` {atomic_smearing} has to be positive")

        self.atomic_smearing = atomic_smearing
        self.sr_cutoff = sr_cutoff
        self.lr_wavelength = lr_wavelength

        # If interior contributions are to be subtracted, also do so for self term
        if subtract_interior:
            subtract_self = True
        self.subtract_self = subtract_self
        self.subtract_interior = subtract_interior

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

    # This function is kept to keep MeshLODE compatible with the broader pytorch
    # infrastructure, which require a "forward" function. We name this function
    # "compute" instead, for compatibility with other COSMO software.
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

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        cell: Union[None, torch.Tensor],
        charges: torch.Tensor,
        neighbor_indices: Union[None, torch.Tensor],
        neighbor_shifts: Union[None, torch.Tensor],
    ) -> torch.Tensor:
        # Check that the realspace cutoff (if provided) is not too large
        # This is because the current implementation is not able to return multiple
        # periodic images of the same atom as a neighbor
        cell_dimensions = torch.linalg.norm(cell, dim=1)
        cutoff_max = torch.min(cell_dimensions) / 2 - 1e-6
        if self.sr_cutoff is not None:
            if self.sr_cutoff > torch.min(cell_dimensions) / 2:
                raise ValueError(f"sr_cutoff {self.sr_cutoff} has to be > {cutoff_max}")

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
            sr_cutoff = cutoff_max
        else:
            sr_cutoff = self.sr_cutoff

        if self.atomic_smearing is None:
            smearing = cutoff_max / 5.0
        else:
            smearing = self.atomic_smearing

        if self.lr_wavelength is None:
            lr_wavelength = 0.5 * smearing
        else:
            lr_wavelength = self.lr_wavelength

        potential_sr = self._compute_sr(
            positions=positions,
            charges=charges,
            cell=cell,
            smearing=smearing,
            sr_cutoff=sr_cutoff,
        )

        potential_lr = self._compute_lr(
            positions=positions,
            charges=charges,
            cell=cell,
            smearing=smearing,
            lr_wavelength=lr_wavelength,
        )

        potential_ewald = potential_sr + potential_lr
        return potential_ewald

    def _generate_kvectors(self, ns: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """
        For a given unit cell, compute all reciprocal space vectors that are used to
        perform sums in the Fourier transformed space.

        Note that this function is different from the function implemented in the
        FourierSpaceConvolution class of the same name, since in this case, we are
        generating the full grid of k-vectors, rather than the one that is adapted
        specifically to be used together with FFT.

        :param ns: torch.tensor of shape ``(3,)`` containing integers
            ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
            z-direction, respectively.
        :param cell: torch.tensor of shape ``(3, 3)`` Tensor specifying the real space
            unit cell of a structure, where cell[i] is the i-th basis vector

        :return: torch.tensor of shape ``(N, 3)`` Contains all reciprocal space vectors
            that will be used during Ewald summation (or related approaches).
            ``k_vectors[i]`` contains the i-th vector, where the order has no special
            significance.
            The total number N of k-vectors is NOT simply nx*ny*nz, and roughly
            corresponds to nx*ny*nz/2 due since the vectors +k and -k can be grouped
            together during summation.
        """
        # Check that the shapes of all inputs are correct
        if ns.shape != (3,):
            raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

        # Define basis vectors of the reciprocal cell
        reciprocal_cell = 2 * torch.pi * cell.inverse().T
        bx = reciprocal_cell[0]
        by = reciprocal_cell[1]
        bz = reciprocal_cell[2]

        # Generate all reciprocal space vectors
        nxs_1d = ns[0] * torch.fft.fftfreq(ns[0], device=ns.device)
        nys_1d = ns[1] * torch.fft.fftfreq(ns[1], device=ns.device)
        nzs_1d = ns[2] * torch.fft.fftfreq(ns[2], device=ns.device)  # real FFT
        nxs, nys, nzs = torch.meshgrid(nxs_1d, nys_1d, nzs_1d, indexing="ij")
        nxs = nxs.flatten().reshape((-1, 1))
        nys = nys.flatten().reshape((-1, 1))
        nzs = nzs.flatten().reshape((-1, 1))
        k_vectors = nxs * bx + nys * by + nzs * bz

        return k_vectors

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
        # Define k-space cutoff from required real-space resolution
        k_cutoff = 2 * torch.pi / lr_wavelength

        # Compute number of times each basis vector of the reciprocal space can be
        # scaled until the cutoff is reached
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_float = k_cutoff * basis_norms / 2 / torch.pi
        ns = torch.ceil(ns_float).long()

        # Generate k-vectors and evaluate
        kvectors = self._generate_kvectors(ns=ns, cell=cell)
        knorm_sq = torch.sum(kvectors**2, dim=1)

        # G(k) is the Fourier transform of the Coulomb potential
        # generated by a Gaussian charge density
        # We remove the singularity at k=0 by explicitly setting its
        # value to be equal to zero. This mathematically corresponds
        # to the requirement that the net charge of the cell is zero.
        # G = 4 * torch.pi * torch.exp(-0.5 * smearing**2 * knorm_sq) / knorm_sq
        G = self.potential.potential_fourier_from_k_sq(knorm_sq, smearing)
        G[0] = self.potential.potential_fourier_at_zero(smearing)

        # Compute the energy using the explicit method that
        # follows directly from the Poisson summation formula.
        # For this, we precompute trigonometric factors for optimization, which leads
        # to N^2 rather than N^3 scaling.
        trig_args = kvectors @ (positions.T)  # shape num_k x num_atoms

        # Reshape charges into suitable form for array/tensor broadcasting
        num_atoms = len(positions)
        if charges.dim() > 1:
            num_channels = charges.shape[1]
            charges_reshaped = (charges.T).reshape(num_channels, 1, num_atoms)
            sum_idx = 2
        else:
            charges_reshaped = charges
            sum_idx = 1

        # Actual computation of trigonometric factors
        cos_all = torch.cos(trig_args)
        sin_all = torch.sin(trig_args)
        cos_summed = torch.sum(cos_all * charges_reshaped, dim=sum_idx)
        sin_summed = torch.sum(sin_all * charges_reshaped, dim=sum_idx)

        # Add up the contributions to compute the potential
        energy = torch.zeros_like(charges)
        for i in range(num_atoms):
            energy[i] += torch.sum(
                G * cos_all[:, i] * cos_summed, dim=sum_idx - 1
            ) + torch.sum(G * sin_all[:, i] * sin_summed, dim=sum_idx - 1)
        energy /= torch.abs(cell.det())

        # Remove self contribution if desired
        # For now, this is the expression for the Coulomb potential p=1
        # TODO: modify to expression for general p
        if subtract_self:
            self_contrib = (
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=cell.device)) / smearing
            )
            energy -= charges * self_contrib

        return energy

    def _compute_sr(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        smearing: torch.Tensor,
        sr_cutoff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the short-range part of the Ewald sum in realspace

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
        :param sr_cutoff: Cutoff radius used for the short-range part of the Ewald sum.

        :returns: torch.tensor of shape `(n_atoms, n_channels)` containing the potential
        at the position of each atom for the `n_channels` independent meshes separately.
        """
        # Get list of neighbors
        struc = Atoms(positions=positions.detach().numpy(), cell=cell, pbc=True)
        atom_is, atom_js, shifts = neighbor_list(
            "ijS", struc, sr_cutoff.item(), self_interaction=False
        )

        # Compute energy
        potential = torch.zeros_like(charges)
        for i, j, shift in zip(atom_is, atom_js, shifts):
            dist = torch.linalg.norm(
                positions[j] - positions[i] + torch.tensor(shift.dot(struc.cell))
            )

            # If the contribution from all atoms within the cutoff is to be subtracted
            # this short-range part will simply use -V_LR as the potential
            if self.subtract_interior:
                potential_bare = -self.potential.potential_lr_from_dist(dist, smearing)
            # In the remaining cases, we simply use the usual V_SR to get the full
            # 1/r^p potential when combined with the long-range part implemented in
            # reciprocal space
            else:
                potential_bare = self.potential.potential_sr_from_dist(dist, smearing)
            potential[i] += charges[j] * potential_bare

        return potential
