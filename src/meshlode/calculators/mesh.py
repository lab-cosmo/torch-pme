from typing import List, Optional

import torch

from meshlode.lib.fourier_convolution import FourierSpaceConvolution
from meshlode.lib.mesh_interpolator import MeshInterpolator

from .calculator_base_periodic import CalculatorBasePeriodic

class MeshPotential(CalculatorBasePeriodic):
    """A specie-wise long-range potential, computed using the particle-mesh Ewald (PME)
    method scaling as O(NlogN) with respect to the number of particles N.

    :param atomic_smearing: Width of the atom-centered Gaussian used to create the
        atomic density.
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically be set
        to half of ``atomic_smearing``.
    :param interpolation_order: Interpolation order for mapping onto the grid, where an
        interpolation order of p corresponds to interpolation by a polynomial of degree
        ``p - 1`` (e.g. ``p = 4`` for cubic interpolation).
    :param subtract_self: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from that atom itself (but not
        the periodic images).
    :param all_types: Optional global list of all atomic types that should be considered
        for the computation. This option might be useful when running the calculation on
        subset of a whole dataset and it required to keep the shape of the output
        consistent. If this is not set the possible atomic types will be determined when
        calling the :meth:`compute()`.

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

    name = "MeshPotential"

    def __init__(
        self,
        atomic_smearing: float,
        mesh_spacing: Optional[float] = None,
        interpolation_order: Optional[int] = 4,
        subtract_self: Optional[bool] = False,
        all_types: Optional[List[int]] = None,
        exponent: Optional[torch.Tensor] = torch.tensor(1., dtype=torch.float64),
    ):
        super().__init__(all_types=all_types, exponent=exponent)

        # Check that all provided values are correct
        if interpolation_order not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_order` from 1 to 5 are allowed")
        if atomic_smearing <= 0:
            raise ValueError(f"`atomic_smearing` {atomic_smearing} has to be positive")

        # If no explicit mesh_spacing is given, set it such that it can resolve
        # the smeared potentials.
        if mesh_spacing is None:
            mesh_spacing = atomic_smearing / 2

        # Store provided parameters
        self.atomic_smearing = atomic_smearing
        self.mesh_spacing = mesh_spacing
        self.interpolation_order = interpolation_order
        self.subtract_self = subtract_self

        # Initilize auxiliary objects
        self.fourier_space_convolution = FourierSpaceConvolution()

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        mesh_spacing: Optional[float] = None,
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
        # Initializations
        n_atoms = len(positions)
        assert positions.shape == (n_atoms, 3)
        assert charges.shape[0] == n_atoms

        assert positions.dtype == cell.dtype and charges.dtype == cell.dtype
        assert positions.device == cell.device and charges.device == cell.device

        
        # Define cutoff in reciprocal space
        if mesh_spacing is None:
            mesh_spacing = self.mesh_spacing
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
