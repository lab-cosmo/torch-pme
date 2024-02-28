from typing import List, Optional, Union

import torch

from meshlode.lib.fourier_convolution import FourierSpaceConvolution
from meshlode.lib.mesh_interpolator import MeshInterpolator
from meshlode.lib.system import System


def _1d_tolist(x: torch.Tensor) -> List[int]:
    """Auxilary function to convert 1d torch tensor to list of integers."""
    result: List[int] = []
    for i in x:
        result.append(i.item())
    return result


def _is_subset(tensor1: torch.tensor, tensor2: torch.tensor) -> bool:
    """Checks wether if all elements of tensor1 are part of tensor2."""
    return torch.all(torch.tensor([i in tensor2 for i in tensor1]))


class MeshPotential(torch.nn.Module):
    """A specie-wise long-range potential.

    :param atomic_smearing: Width of the atom-centered gaussian used to create the
        atomic density.
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically
        be set to half of ``atomic_smearing``
    :param interpolation_order: Interpolation order for mapping onto the grid, where an
        interpolation order of p corresponds to interpolation by a polynomial of degree
        ``p-1`` (e.g. ``p=4`` for cubic interpolation).
    :param subtract_self: bool. If set to true, subtract from the features of an atom
        the contributions to the potential arising from that atom itself (but not the
        periodic images).
    :param all_atomic_numbers: Optional global list of all atomic numbers that should be
        considered for the computation. This option might be useful when running the
        calculation on subset of a whole dataset and it required to keep the shape of
        the output consistent. If this is not set the possible number of species will be
        determined when calling the :meth:`compute()`.

    Example
    -------
    >>> import torch
    >>> from meshlode.lib import System
    >>> from meshlode import MeshPotential

    Define simple example structure having the CsCl (Cesium Chloride) structure

    >>> positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> atomic_numbers = torch.tensor([55, 17])  # Cs and Cl
    >>> system = System(species=atomic_numbers, positions=positions, cell=torch.eye(3))

    Compute features

    >>> MP = MeshPotential(atomic_smearing=0.2, mesh_spacing=0.1, interpolation_order=4)
    >>> MP.compute(system)
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
        all_atomic_numbers: Optional[List[int]] = None,
    ):
        super().__init__()

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

        if all_atomic_numbers is None:
            self.all_atomic_numbers = None
        else:
            self.all_atomic_numbers = _1d_tolist(
                torch.unique(torch.tensor(all_atomic_numbers))
            )

        # Initilize auxiliary objects
        self.fourier_space_convolution = FourierSpaceConvolution()

    # This function is kept to keep MeshLODE compatible with the broader pytorch
    # infrastructure, which require a "forward" function. We name this function
    # "compute" instead, for compatibility with other COSMO software.
    def forward(
        self,
        systems: Union[List[System], System],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """forward just calls :py:meth:`CalculatorModule.compute`"""
        return self.compute(systems=systems)

    def compute(
        self,
        systems: Union[List[System], System],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute the potential at the position of each atom for all provided systems.

        :param systems: single System or list of Systems on which to run the
            calculation. If any of the systems' ``positions`` or ``cell`` has
            ``requires_grad`` set to :py:obj:`True`, then the corresponding gradients
            are computed and registered as a custom node in the computational graph, to
            allow backward propagation of the gradients later.

        :return: List of torch Tensors containing the potentials for all frames and all
            atoms. Each tensor in the list is of shape (n_atoms,n_species), where
            n_species is the number of species in all systems combined. If the input was
            a single system only a single torch tensor with the potentials is returned.

            IMPORTANT: If multiple species are present, the different "species-channels"
            are ordered according to atomic number. For example, if a structure contains
            a water molecule with atoms 0, 1, 2 being of species O, H, H, then for this
            system, the feature tensor will be of shape (3, 2) = (``n_atoms``,
            ``n_species``), where ``features[0, 0]`` is the potential at the position of
            the Oxygen atom (atom 0, first index) generated by the HYDROGEN(!) atoms,
            while ``features[0,1]`` is the potential at the position of the Oxygen atom
            generated by the Oxygen atom(s).
        """
        # Make sure that the compute function also works if only a single frame is
        # provided as input (for convenience of users testing out the code)
        if not isinstance(systems, list):
            systems = [systems]

        atomic_numbers = self._get_atomic_numbers(systems)
        n_species = len(atomic_numbers)

        potentials = []
        for system in systems:
            # One-hot encoding of charge information
            n_atoms = len(system)
            species = system.species
            charges = torch.zeros((n_atoms, n_species), dtype=torch.float)
            for i_specie, atomic_number in enumerate(atomic_numbers):
                charges[species == atomic_number, i_specie] = 1.0

            # Compute the potentials
            potentials.append(
                self._compute_single_frame(system.cell, system.positions, charges)
            )

        if len(systems) == 1:
            return potentials[0]
        else:
            return potentials

    def _get_atomic_numbers(self, systems: List[System]) -> List[int]:
        """Extract all species/atomic_numbers from the list of systems."""
        all_species = [system.species for system in systems]
        all_species = torch.hstack(all_species)
        atomic_numbers_requested = _1d_tolist(torch.unique(all_species))

        if self.all_atomic_numbers is not None:
            if not _is_subset(atomic_numbers_requested, self.all_atomic_numbers):
                raise ValueError(
                    f"Global list of atomic numbers {self.all_atomic_numbers} does not "
                    "contain all atomic numbers for the provided systems "
                    f"{atomic_numbers_requested}."
                )
            return self.all_atomic_numbers
        else:
            return atomic_numbers_requested

    def _compute_single_frame(
        self,
        cell: torch.Tensor,
        positions: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the "electrostatic" potential at the position of all atoms in a
        structure.

        :param cell: torch.tensor of shape `(3, 3)`. Describes the unit cell of the
            structure, where cell[i] is the i-th basis vector.
        :param positions: torch.tensor of shape (n_atoms, 3). Contains the Cartesian
            coordinates of the atoms. The implementation also works if the positions
            are not contained within the unit cell.
        :param charges: torch.tensor of shape `(n_atoms, n_channels)`. In the simplest
            case, this would be a tensor of shape (n_atoms, 1) where charges[i,0] is the
            charge of atom i. More generally, the potential for the same atom positions
            is computed for n_channels independent meshes, and one can specify the
            "charge" of each atom on each of the meshes independently. For standard LODE
            that treats all atomic species separately, one example could be: If n_atoms
            = 4 and the species are [Na, Cl, Cl, Na], one could set n_channels=2 and use
            the one-hot encoding charges = torch.tensor([[1,0],[0,1],[0,1],[1,0]]) for
            the charges. This would then separately compute the "Na" potential and "Cl"
            potential. Subtracting these from each other, one could recover the more
            standard electrostatic potential in which Na and Cl have charges of +1 and
            -1, respectively.

        :returns: torch.tensor of shape `(n_atoms, n_channels)` containing the potential
        at the position of each atom for the `n_channels` independent meshes separately.
        """
        # Initializations
        n_atoms = len(positions)
        assert positions.shape == (n_atoms, 3)
        assert charges.shape[0] == n_atoms

        # Define cutoff in reciprocal space
        k_cutoff = 2 * torch.pi / self.mesh_spacing

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
                torch.sqrt(torch.tensor(2.0 / torch.pi)) / self.atomic_smearing
            )
            interpolated_potential -= charges * self_contrib

        return interpolated_potential
