from typing import Dict, List, Union

import torch


try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
except ImportError:
    raise ImportError(
        "metatensor.torch is required for meshlode.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    )

from metatensor.torch.atomistic import System

from .. import calculators


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
    >>> atomic_types = torch.tensor([55, 17])  # Cs and Cl
    >>> frame = System(species=atomic_types, positions=positions, cell=torch.eye(3))

    Compute features

    >>> MP = MeshPotential(atomic_smearing=0.2, mesh_spacing=0.1, interpolation_order=4)
    >>> features = MP.compute(frame)

    All species combinations

    >>> features.keys
    Labels(
        center_type  neighbor_type
            17            17
            17            55
            55            17
            55            55
    )

    The Cl-potential at the position of the Cl atom

    >>> block_ClCl = features.block({"center_type": 17, "neighbor_type": 17})
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
            tensormap are "center_type" and "neighbor_type".
        """
        # Make sure that the compute function also works if only a single frame is
        # provided as input (for convenience of users testing out the code)
        if not isinstance(systems, list):
            systems = [systems]

        atomic_types = self._get_atomic_types(systems)
        n_species = len(atomic_types)

        # Initialize dictionary for sparse storage of the features Generate a dictionary
        # to map atomic species to array indices In general, the species are sorted
        # according to atomic number and assigned the array indices 0, 1, 2,...
        # Example: for H2O: `H` is mapped to `0` and `O` is mapped to `1`.
        n_species_sq = n_species * n_species
        feat_dic: Dict[int, List[torch.Tensor]] = {a: [] for a in range(n_species_sq)}

        for system in systems:
            # One-hot encoding of charge information
            n_atoms = len(system)
            species = system.species
            charges = torch.zeros((n_atoms, n_species), dtype=torch.float)
            for i_specie, atomic_type in enumerate(atomic_types):
                charges[species == atomic_type, i_specie] = 1.0

            # Compute the potentials
            potential = self._compute_single_frame(
                system.positions, charges, system.cell
            )

            # Reorder data into Metatensor format
            for spec_center, at_num_center in enumerate(atomic_types):
                for spec_neighbor in range(len(atomic_types)):
                    a_pair = spec_center * n_species + spec_neighbor
                    feat_dic[a_pair] += [
                        potential[species == at_num_center, spec_neighbor]
                    ]

        # Assemble all computed potential values into TensorBlocks for each combination
        # of center_type and neighbor_type
        blocks: List[TensorBlock] = []
        for keys, values in feat_dic.items():
            spec_center = atomic_types[keys // n_species]

            # Generate the Labels objects for the samples and properties of the
            # TensorBlock.
            values_samples: List[List[int]] = []
            for i_frame, system in enumerate(systems):
                for i_atom in range(len(system)):
                    if system.species[i_atom] == spec_center:
                        values_samples.append([i_frame, i_atom])

            samples_vals_tensor = torch.tensor(values_samples, dtype=torch.int32)

            # If no atoms are found that match the species pair `samples_vals_tensor`
            # will be empty. We have to reshape the empty tensor to be a valid input for
            # `Labels`.
            if len(samples_vals_tensor) == 0:
                samples_vals_tensor = samples_vals_tensor.reshape(-1, 2)

            labels_samples = Labels(["system", "atom"], samples_vals_tensor)
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
        for spec_center in atomic_types:
            for spec_neighbor in atomic_types:
                key_values.append(torch.tensor([spec_center, spec_neighbor]))
        key_values = torch.vstack(key_values)
        labels_keys = Labels(["center_type", "neighbor_type"], key_values)

        return TensorMap(keys=labels_keys, blocks=blocks)
