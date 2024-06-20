from typing import List, Optional

import torch

# extra imports for neighbor list
from ase import Atoms
from ase.neighborlist import neighbor_list

from meshlode.lib.mesh_interpolator import MeshInterpolator

from .calculator_base import default_exponent

# from .mesh import MeshPotential
from .calculator_base_periodic import CalculatorBasePeriodic


class MeshEwaldPotential(CalculatorBasePeriodic):
    """A specie-wise long-range potential computed using a mesh-based Ewald method,
    scaling as O(NlogN) with respect to the number of particles N used as a reference
    to test faster implementations.

    :param all_types: Optional global list of all atomic types that should be considered
        for the computation. This option might be useful when running the calculation on
        subset of a whole dataset and it required to keep the shape of the output
        consistent. If this is not set the possible atomic types will be determined when
        calling the :meth:`compute()`.
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
    """

    name = "MeshEwaldPotential"

    def __init__(
        self,
        all_types: Optional[List[int]] = None,
        exponent: Optional[torch.Tensor] = default_exponent,
        sr_cutoff: Optional[torch.Tensor] = None,
        atomic_smearing: Optional[float] = None,
        mesh_spacing: Optional[float] = None,
        subtract_self: Optional[bool] = True,
        interpolation_order: Optional[int] = 4,
        subtract_interior: Optional[bool] = False,
    ):
        super().__init__(all_types=all_types, exponent=exponent)

        # Check that all provided values are correct
        if interpolation_order not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_order` from 1 to 5 are allowed")
        if atomic_smearing is not None and atomic_smearing <= 0:
            raise ValueError(f"`atomic_smearing` {atomic_smearing} has to be positive")

        # Store provided parameters
        self.atomic_smearing = atomic_smearing
        self.mesh_spacing = mesh_spacing
        self.interpolation_order = interpolation_order
        self.sr_cutoff = sr_cutoff

        # If interior contributions are to be subtracted, also do so for self term
        if subtract_interior:
            subtract_self = True
        self.subtract_self = subtract_self
        self.subtract_interior = subtract_interior

    def _generate_kvectors(self, ns: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """
        For a given unit cell, compute all reciprocal space vectors that are used to
        perform sums in the Fourier transformed space.

        :param ns: torch.tensor of shape ``(3,)``
            ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
            z-direction, respectively. For faster performance during the Fast Fourier
            Transform (FFT) it is recommended to use values of nx, ny and nz that are
            powers of 2.
        :param cell: torch.tensor of shape ``(3, 3)`` Tensor specifying the real space
            unit cell of a structure, where cell[i] is the i-th basis vector

        :return: torch.tensor of shape ``(N, 3)`` Contains all reciprocal space vectors
            that will be used during Ewald summation (or related approaches).
            ``k_vectors[i]`` contains the i-th vector, where the order has no special
            significance.
        """
        if ns.device != cell.device:
            raise ValueError(
                f"`ns` and `cell` are not on the same device, got {ns.device} and "
                f"{cell.device}."
            )

        if ns.shape != (3,):
            raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

        if cell.shape != (3, 3):
            raise ValueError(
                f"cell of shape {list(cell.shape)} should be of shape (3, 3)"
            )

        # Define basis vectors of the reciprocal cell
        reciprocal_cell = 2 * torch.pi * cell.inverse().T
        bx = reciprocal_cell[0]
        by = reciprocal_cell[1]
        bz = reciprocal_cell[2]

        # Generate all reciprocal space vectors
        nxs_1d = ns[0] * torch.fft.fftfreq(ns[0], device=ns.device)
        nys_1d = ns[1] * torch.fft.fftfreq(ns[1], device=ns.device)
        nzs_1d = ns[2] * torch.fft.rfftfreq(ns[2], device=ns.device)  # real FFT
        nxs, nys, nzs = torch.meshgrid(nxs_1d, nys_1d, nzs_1d, indexing="ij")
        nxs = nxs.reshape((int(ns[0]), int(ns[1]), len(nzs_1d), 1))
        nys = nys.reshape((int(ns[0]), int(ns[1]), len(nzs_1d), 1))
        nzs = nzs.reshape((int(ns[0]), int(ns[1]), len(nzs_1d), 1))
        k_vectors = nxs * bx + nys * by + nzs * bz

        return k_vectors

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

        :returns: torch.tensor of shape `(n_atoms, n_channels)` containing the potential
        at the position of each atom for the `n_channels` independent meshes separately.
        """
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
            neighbor_shifts=neighbor_shifts
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
        kvectors = self._generate_kvectors(ns=ns, cell=cell)
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

    def _compute_sr(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        smearing: torch.Tensor,
        sr_cutoff: torch.Tensor,
        neighbor_indices: Optional[torch.Tensor] = None,
        neighbor_shifts: Optional[torch.Tensor] = None,
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
        if neighbor_indices is None:
            # Get list of neighbors
            struc = Atoms(positions=positions.detach().numpy(), cell=cell, pbc=True)
            atom_is, atom_js, shifts = neighbor_list(
                "ijS", struc, sr_cutoff.item(), self_interaction=False
            )
        else:
            atom_is = neighbor_indices[:,0]
            atom_js = neighbor_indices[:,1]
            shifts = neighbor_shifts.T
            

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
