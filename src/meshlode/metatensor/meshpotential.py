from typing import Dict, List, Union

import torch


try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for meshlode.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    )


from .. import calculators


# We are breaking the Liskov substitution principle here by changing the signature of
# "compute" compated to the supertype of "MeshPotential".
# mypy: disable-error-code="override"


class MeshPotential(calculators.MeshPotential):
    """An (atomic) type wise long range potential.

    Refer to :class:`meshlode.MeshPotential` for full documentation.

    Example
    -------
    >>> import torch
    >>> from metatensor.torch.atomistic import System
    >>> from meshlode.metatensor import MeshPotential

    Define simple example structure having the CsCl (Cesium Chloride) structure

    >>> types = torch.tensor([55, 17])  # Cs and Cl
    >>> positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> cell = torch.eye(3)
    >>> system = System(types=types, positions=positions, cell=cell)

    Compute features

    >>> MP = MeshPotential(atomic_smearing=0.2, mesh_spacing=0.1, interpolation_order=4)
    >>> features = MP.compute(system)

    All (atomic) type combinations

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
        """Compute potential for all provided ``systems``.

        All ``systems`` must have the same ``dtype`` and the same ``device``. If each
        system contains a custom data field `charges` the potential will be calculated
        for each "charges-channel". The number of `charges-channels` must be same in all
        ``systems``. If no "explicit" charges are set the potential will be calculated
        for each "types-channels".

        Refer to :meth:`meshlode.MeshPotential.compute()` for additional details on how
        "charges-channel" and "types-channels" are computed.

        :param systems: single System or list of
            :py:class:`metatensor.torch.atomisic.System` on which to run the
            calculations.

        :return: TensorMap containing the potential of all types. The keys of the
            TensorMap are "center_type" and "neighbor_type" if no charges are asociated
            with
        """
        # Make sure that the compute function also works if only a single frame is
        # provided as input (for convenience of users testing out the code)
        if not isinstance(systems, list):
            systems = [systems]

        if len(systems) > 1:
            for system in systems[1:]:
                if system.dtype != systems[0].dtype:
                    raise ValueError(
                        "`dtype` of all systems must be the same, got "
                        f"{system.dtype} and {systems[0].dtype}`"
                    )

                if system.device != systems[0].device:
                    raise ValueError(
                        "`device` of all systems must be the same, got "
                        f"{system.device} and {systems[0].device}`"
                    )

        dtype = systems[0].positions.dtype
        device = systems[0].positions.device

        requested_types = self._get_requested_types(
            [system.types for system in systems]
        )
        n_types = len(requested_types)

        has_charges = torch.tensor(["charges" in s.known_data() for s in systems])
        all_charges = torch.all(has_charges)
        any_charges = torch.any(has_charges)

        if any_charges and not all_charges:
            raise ValueError("`systems` do not consistently contain `charges` data")
        if all_charges:
            use_explicit_charges = True
            n_charges_channels = systems[0].get_data("charges").values.shape[1]
            spec_channels = list(range(n_charges_channels))
            key_names = ["center_type", "charges_channel"]

            for i_system, system in enumerate(systems):
                n_channels = system.get_data("charges").values.shape[1]
                if n_channels != n_charges_channels:
                    raise ValueError(
                        f"number of charges-channels in system index {i_system} "
                        f"({n_channels}) is inconsistent with first system "
                        f"({n_charges_channels})"
                    )
        else:
            # Use one hot encoded type channel per species for charges channel
            use_explicit_charges = False
            n_charges_channels = n_types
            spec_channels = requested_types
            key_names = ["center_type", "neighbor_type"]

        # Initialize dictionary for TensorBlock storage.
        #
        # If `use_explicit_charges=False`, the blocks are sorted according to the
        # (integer) center_type and neighbor_type. Blocks are assigned the array indices
        # 0, 1, 2,... Example: for H2O: `H` is mapped to `0` and `O` is mapped to `1`.
        #
        # For `use_explicit_charges=True` the blocks are stored according to the
        # center_type and charge_channel
        n_blocks = n_types * n_charges_channels
        feat_dic: Dict[int, List[torch.Tensor]] = {a: [] for a in range(n_blocks)}

        for system in systems:
            if use_explicit_charges:
                charges = system.get_data("charges").values
            else:
                # One-hot encoding of charge information
                charges = self._one_hot_charges(
                    system.types, requested_types, n_types, dtype, device
                )

            # Compute the potentials
            potential = self._compute_single_system(
                system.positions, charges, system.cell
            )

            # Reorder data into metatensor format
            for spec_center, at_num_center in enumerate(requested_types):
                for spec_channel in range(len(spec_channels)):
                    a_pair = spec_center * n_charges_channels + spec_channel
                    feat_dic[a_pair] += [
                        potential[system.types == at_num_center, spec_channel]
                    ]

        # Assemble all computed potential values into TensorBlocks for each combination
        # of center_type and neighbor_type/charge_channel
        blocks: List[TensorBlock] = []
        for keys, values in feat_dic.items():
            spec_center = requested_types[keys // n_charges_channels]

            # Generate the Labels objects for the samples and properties of the
            # TensorBlock.
            values_samples: List[List[int]] = []
            for i_frame, system in enumerate(systems):
                for i_atom in range(len(system)):
                    if system.types[i_atom] == spec_center:
                        values_samples.append([i_frame, i_atom])

            samples_vals_tensor = torch.tensor(
                values_samples, dtype=torch.int32, device=device
            )

            # If no atoms are found that match the types pair `samples_vals_tensor`
            # will be empty. We have to reshape the empty tensor to be a valid input for
            # `Labels`.
            if len(samples_vals_tensor) == 0:
                samples_vals_tensor = samples_vals_tensor.reshape(-1, 2)

            labels_samples = Labels(["system", "atom"], samples_vals_tensor)
            labels_properties = Labels(
                ["potential"], torch.tensor([[0]], device=device)
            )

            block = TensorBlock(
                samples=labels_samples,
                components=[],
                properties=labels_properties,
                values=torch.hstack(values).reshape((-1, 1)),
            )

            blocks.append(block)

        assert len(blocks) == n_blocks

        # Generate TensorMap from TensorBlocks by defining suitable keys
        key_values: List[torch.Tensor] = []
        for spec_center in requested_types:
            for spec_channel in spec_channels:
                key_values.append(
                    torch.tensor([spec_center, spec_channel], device=device)
                )
        key_values = torch.vstack(key_values)

        labels_keys = Labels(key_names, key_values)

        return TensorMap(keys=labels_keys, blocks=blocks)
