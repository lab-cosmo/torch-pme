"""
Available Calculators
=====================

Below is a list of all calculators available. Calculators are the core of MeshLODE and
are algorithms for transforming Cartesian coordinates into representations suitable for
machine learning.

Our calculator API follows the `rascaline <https://luthaf.fr/rascaline>`_ API and coding
guidelines to promote usability and interoperability with existing workflows.
"""
from typing import Dict, List, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from meshlode.fourier_convolution import FourierSpaceConvolution
from meshlode.mesh_interpolator import MeshInterpolator
from meshlode.system import System


def _my_1d_tolist(x: torch.Tensor):
    """Auxilary function to convert torch tensor to list of integers"""
    result: List[int] = []
    for i in x:
        result.append(i.item())
    return result


class MeshPotential(torch.nn.Module):
    """A species wise long range potential.

    :param atomic_gaussian_width: Width of the atom-centered gaussian used to create the
        atomic density.
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis.
    :param interpolation_order: Interpolation order for mapping onto the grid, where an
        interpolation order of p corresponds to interpolation by a polynomial of degree
        ``p-1`` (e.g. ``p=4`` for cubic interpolation).
    :param subtract_self: bool. If set to true, subtract from the features of an atom
        the contributions to the potential arising from that atom itself (but not the
        periodic images).

    Example
    -------

    >>> import torch
    >>> from meshlode import MeshPotential, System

    Define simple example structure having the CsCl structure

    >>> positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> atomic_numbers = torch.tensor([55, 17])  # Cs and Cl
    >>> frame = System(species=atomic_numbers, positions=positions, cell=torch.eye(3))

    Compute features

    >>> MP = MeshPotential(
    ...     atomic_gaussian_width=0.2, mesh_spacing=0.1, interpolation_order=4
    ... )
    >>> features = MP.compute(frame)

    All species combinations

    >>> features.keys
    Labels(
        species_center  species_neighbor
              17               17
              17               55
              55               17
              55               55
    )
    >>> block_ClCl = features.block({"species_center": 17, "species_neighbor": 17})

    The Cl-potential at the position of the Cl atom

    >>> block_ClCl.values
    tensor([[1.3755]])

    """

    name = "MeshPotential"

    def __init__(
        self,
        atomic_gaussian_width: float,
        mesh_spacing: float = 0.2,
        interpolation_order: float = 4,
        subtract_self: bool = False,
    ):
        super().__init__()

        self.atomic_gaussian_width = atomic_gaussian_width
        self.mesh_spacing = mesh_spacing
        self.interpolation_order = interpolation_order
        self.subtract_self = subtract_self

    # This function is kept to keep MeshLODE compatible with the broader pytorch
    # infrastructure, which require a "forward" function. We name this function
    # "compute" instead, for compatibility with other COSMO software.
    def forward(
        self,
        systems: Union[List[System], System],
    ) -> TensorMap:
        """forward just calls :py:meth:`CalculatorModule.compute`"""
        res = self.compute(frames=systems)
        return res
        # return 0.

    def compute(
        self,
        frames: Union[List[System], System],
    ) -> TensorMap:
        """Compute the potential at the position of each atom for all Systems provided
        in "frames".

        :param frames: single System or list of Systems on which to run the
            calculation. If any of the systems' ``positions`` or ``cell`` has
            ``requires_grad`` set to :py:obj:`True`, then the corresponding gradients
            are computed and registered as a custom node in the computational graph, to
            allow backward propagation of the gradients later.

        :return: TensorMap containing the potential of all atoms. The keys of the
            tensormap are "species_center" and "species_neighbor".
        """
        # Make sure that the compute function also works if only a single frame is
        # provided as input (for convenience of users testing out the code)
        if not isinstance(frames, list):
            frames = [frames]

        # Generate a dictionary to map atomic species to array indices
        # In general, the species are sorted according to atomic number
        # and assigned the array indices 0, 1, 2,...
        # Example: for H2O: H is mapped to 0 and O is mapped to 1.
        all_species = []
        n_atoms_tot = 0
        for frame in frames:
            n_atoms_tot += len(frame)
            all_species.append(frame.species)
        all_species = torch.hstack(all_species)
        atomic_numbers = _my_1d_tolist(torch.unique(all_species))
        n_species = len(atomic_numbers)

        # Initialize dictionary for sparse storage of the features
        n_species_sq = n_species * n_species
        feat_dic: Dict[int, List[torch.Tensor]] = {a: [] for a in range(n_species_sq)}

        for frame in frames:
            # One-hot encoding of charge information
            n_atoms = len(frame)
            species = frame.species
            charges = torch.zeros((n_atoms, n_species), dtype=torch.float)
            for i_specie, atomic_number in enumerate(atomic_numbers):
                charges[species == atomic_number, i_specie] = 1.0

            # Compute the potentials
            potential = self._compute_single_frame(frame.cell, frame.positions, charges)

            # Reorder data into Metatensor format
            for spec_center, at_num_center in enumerate(atomic_numbers):
                for spec_neighbor in range(len(atomic_numbers)):
                    a_pair = spec_center * n_species + spec_neighbor
                    feat_dic[a_pair] += [
                        potential[species == at_num_center, spec_neighbor]
                    ]

        # Assemble all computed potential values into TensorBlocks for each combination
        # of species_center and species_neighbor
        blocks: List[TensorBlock] = []
        for keys, values in feat_dic.items():
            spec_center = atomic_numbers[keys // n_species]

            # Generate the Labels objects for the samples and properties of the
            # TensorBlock.
            samples_vals: List[List[int]] = []
            for i_frame, frame in enumerate(frames):
                for i_atom in range(len(frame)):
                    if frame.species[i_atom] == spec_center:
                        samples_vals.append([i_frame, i_atom])
            samples_vals_tensor = torch.tensor((samples_vals), dtype=torch.int32)
            labels_samples = Labels(["structure", "center"], samples_vals_tensor)

            labels_properties = Labels(["potential"], torch.tensor([[0]]))

            block = TensorBlock(
                samples=labels_samples,
                components=[],
                properties=labels_properties,
                values=torch.hstack(values).reshape((-1, 1)),
            )

            blocks.append(block)

        # Generate TensorMap from TensorBlocks by defining suitable keys
        key_values: List[torch.Tensor] = []
        for spec_center in atomic_numbers:
            for spec_neighbor in atomic_numbers:
                key_values.append(torch.tensor([spec_center, spec_neighbor]))
        key_values = torch.vstack(key_values)
        labels_keys = Labels(["species_center", "species_neighbor"], key_values)

        return TensorMap(keys=labels_keys, blocks=blocks)

    def _compute_single_frame(
        self,
        cell: torch.Tensor,
        positions: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the "electrostatic" potential at the position of all atoms in a
        structure.

        :param cell: torch.tensor of shape (3,3). Describes the unit cell of the
            structure, where cell[i] is the i-th basis vector.
        :param positions: torch.tensor of shape (n_atoms, 3). Contains the Cartesian
            coordinates of the atoms. The implementation also works if the positions
            are not contained within the unit cell.
        :param charges: torch.tensor of shape (n_atoms, n_channels). In the simplest
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

        :returns: torch.tensor of shape (n_atoms, n_channels) containing the potential
        at the position of each atom for the n_channels independent meshes separately.
        """
        smearing = self.atomic_gaussian_width
        mesh_resolution = self.mesh_spacing
        interpolation_order = self.interpolation_order

        # Initializations
        n_atoms = len(positions)
        assert positions.shape == (n_atoms, 3)
        assert charges.shape[0] == n_atoms

        # Define k-vectors
        if mesh_resolution is None:
            k_cutoff = 2 * torch.pi / (smearing / 2)
        else:
            k_cutoff = 2 * torch.pi / mesh_resolution

        # Compute number of times each basis vector of the
        # reciprocal space can be scaled until the cutoff
        # is reached
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_approx = k_cutoff * basis_norms / 2 / torch.pi
        ns_actual_approx = 2 * ns_approx + 1  # actual number of mesh points
        ns = 2 ** torch.ceil(torch.log2(ns_actual_approx)).long()  # [nx, ny, nz]

        # Step 1: Smear particles onto mesh
        MI = MeshInterpolator(cell, ns, interpolation_order=interpolation_order)
        MI.compute_interpolation_weights(positions)
        rho_mesh = MI.points_to_mesh(particle_weights=charges)

        # Step 2: Perform Fourier space convolution (FSC)
        FSC = FourierSpaceConvolution(cell)
        potential_mesh = FSC.compute(rho_mesh, potential_exponent=1, smearing=smearing)

        # Step 3: Back interpolation
        interpolated_potential = MI.mesh_to_points(potential_mesh)

        # Remove self contribution
        if self.subtract_self:
            self_contrib = torch.sqrt(torch.tensor(2.0 / torch.pi)) / smearing
            interpolated_potential -= charges * self_contrib

        return interpolated_potential
