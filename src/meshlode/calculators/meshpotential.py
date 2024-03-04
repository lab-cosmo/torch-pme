from typing import List, Optional, Union

import torch

from meshlode.lib.fourier_convolution import FourierSpaceConvolution
from meshlode.lib.mesh_interpolator import MeshInterpolator


def _1d_tolist(x: torch.Tensor) -> List[int]:
    """Auxilary function to convert 1d torch tensor to list of integers."""
    result: List[int] = []
    for i in x:
        result.append(i.item())
    return result


def _is_subset(tensor1: List[int], tensor2: List[int]) -> bool:
    """Checks wether if all elements of tensor1 are part of tensor2."""
    return torch.all(torch.tensor([i in tensor2 for i in tensor1]))


class MeshPotential(torch.nn.Module):
    """A specie-wise long-range potential.

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

        if all_types is None:
            self.all_types = None
        else:
            self.all_types = _1d_tolist(torch.unique(torch.tensor(all_types)))

        # Initilize auxiliary objects
        self.fourier_space_convolution = FourierSpaceConvolution()

    # This function is kept to keep MeshLODE compatible with the broader pytorch
    # infrastructure, which require a "forward" function. We name this function
    # "compute" instead, for compatibility with other COSMO software.
    def forward(
        self,
        types: Union[List[torch.Tensor], torch.Tensor],
        positions: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """forward just calls :py:meth:`CalculatorModule.compute`"""
        return self.compute(types=types, positions=positions, cell=cell)

    def compute(
        self,
        types: Union[List[torch.Tensor], torch.Tensor],
        positions: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
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
        # make sure compute function works if only a single tensor are provided as input
        if not isinstance(types, list):
            types = [types]
        if not isinstance(positions, list):
            positions = [positions]
        if not isinstance(cell, list):
            cell = [cell]

        for types_single, positions_single, cell_single in zip(types, positions, cell):
            if len(types_single.shape) != 1:
                raise ValueError(
                    "each `types` must be a 1 dimensional tensor, got at least "
                    f"one tensor with {len(types_single.shape)} dimensions"
                )

            if positions_single.shape != (len(types_single), 3):
                raise ValueError(
                    "each `positions` must be a (n_types x 3) tensor, got at least "
                    f"one tensor with shape {list(positions_single.shape)}"
                )

            if cell_single.shape != (3, 3):
                raise ValueError(
                    "each `cell` must be a (3 x 3) tensor, got at least "
                    f"one tensor with shape {list(cell_single.shape)}"
                )

            if cell_single.dtype != positions_single.dtype:
                raise ValueError(
                    "`cell` must be have the same dtype as `positions`, got "
                    f"{cell_single.dtype} and {positions_single.dtype}"
                )

            if (
                positions_single.device != types_single.device
                or cell_single.device != types_single.device
            ):
                raise ValueError(
                    "`types`, `positions`, and `cell` must be on the same device, got "
                    f"{types_single.device}, {positions_single.device} and "
                    f"{cell_single.device}."
                )

        # We don't require and test that all dtypes and devices are consistent if a list
        # of inputs. Each "frame" is processed independently.

        requested_types = self._get_requested_types(types)
        n_types = len(requested_types)

        potentials = []
        for types_single, positions_single, cell_single in zip(types, positions, cell):
            # One-hot encoding of charge information
            charges = torch.zeros(
                (len(types_single), n_types),
                dtype=positions_single.dtype,
                device=positions_single.device,
            )
            for i_type, atomic_type in enumerate(requested_types):
                charges[types_single == atomic_type, i_type] = 1.0

            # Compute the potentials
            potentials.append(
                self._compute_single_system(
                    positions=positions_single, charges=charges, cell=cell_single
                )
            )

        if len(types) == 1:
            return potentials[0]
        else:
            return potentials

    def _get_requested_types(self, types: List[torch.Tensor]) -> List[int]:
        """Extract a list of all unique and present types from the list of types."""
        all_types = torch.hstack(types)
        types_requested = _1d_tolist(torch.unique(all_types))

        if self.all_types is not None:
            if not _is_subset(types_requested, self.all_types):
                raise ValueError(
                    f"Global list of types {self.all_types} does not contain all "
                    f"types for the provided systems {types_requested}."
                )
            return self.all_types
        else:
            return types_requested

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
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
                torch.sqrt(
                    torch.tensor(
                        2.0 / torch.pi, dtype=positions.dtype, device=positions.device
                    ),
                )
                / self.atomic_smearing
            )
            interpolated_potential -= charges * self_contrib

        return interpolated_potential
