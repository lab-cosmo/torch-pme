from typing import Dict, List, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from meshlode.lib.system import System

from .. import calculators
from ..calculators.meshpotential import _my_1d_tolist


class MeshPotential(calculators.MeshPotential):
    """A species wise long range potential.

    Refer to :class:`meshlode.MeshPotential` for full documentation.

    Example
    -------

    >>> import torch
    >>> from meshlode.lib import System
    >>> from meshlode.metatensor import MeshPotential

    Define simple example structure having the CsCl (Cesium Chloride) structure

    >>> positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> atomic_numbers = torch.tensor([55, 17])  # Cs and Cl
    >>> frame = System(species=atomic_numbers, positions=positions, cell=torch.eye(3))

    Compute features

    >>> MP = MeshPotential(atomic_smearing=0.2, mesh_spacing=0.1, interpolation_order=4)
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

    The Cl-potential at the position of the Cl atom

    >>> block_ClCl = features.block({"species_center": 17, "species_neighbor": 17})
    >>> block_ClCl.values
    tensor([[1.3755]])

    """

    def forward(
        self,
        systems: Union[List[System], System],
    ) -> TensorMap:
        """forward just calls :py:meth:`CalculatorModule.compute`"""
        return self.compute(systems=systems)

    def compute(
        self,
        systems: Union[List[System], System],
    ) -> TensorMap:
        """Compute the potential at the position of each atom for all Systems provided
        in "frames".

        :param systems: single System or list of Systems on which to run the
            calculation. If any of the systems' ``positions`` or ``cell`` has
            ``requires_grad`` set to :py:obj:`True`, then the corresponding gradients
            are computed and registered as a custom node in the computational graph, to
            allow backward propagation of the gradients later.

        :return: TensorMap containing the potential of all atoms. The keys of the
            tensormap are "species_center" and "species_neighbor".
        """
        # Make sure that the compute function also works if only a single frame is
        # provided as input (for convenience of users testing out the code)
        if not isinstance(systems, list):
            systems = [systems]

        # Generate a dictionary to map atomic species to array indices
        # In general, the species are sorted according to atomic number
        # and assigned the array indices 0, 1, 2,...
        # Example: for H2O: H is mapped to 0 and O is mapped to 1.
        all_species = []
        n_atoms_tot = 0
        for system in systems:
            n_atoms_tot += len(system)
            all_species.append(system.species)
        all_species = torch.hstack(all_species)
        atomic_numbers = _my_1d_tolist(torch.unique(all_species))
        n_species = len(atomic_numbers)

        # Initialize dictionary for sparse storage of the features
        n_species_sq = n_species * n_species
        feat_dic: Dict[int, List[torch.Tensor]] = {a: [] for a in range(n_species_sq)}

        for system in systems:
            # One-hot encoding of charge information
            n_atoms = len(system)
            species = system.species
            charges = torch.zeros((n_atoms, n_species), dtype=torch.float)
            for i_specie, atomic_number in enumerate(atomic_numbers):
                charges[species == atomic_number, i_specie] = 1.0

            # Compute the potentials
            potential = self._compute_single_frame(
                system.cell, system.positions, charges
            )

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
            for i_frame, system in enumerate(systems):
                for i_atom in range(len(system)):
                    if system.species[i_atom] == spec_center:
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
