from typing import List, Optional, Tuple, Union

import torch


try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for torchpme.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    )


class CalculatorBaseMetatensor(torch.nn.Module):
    """Base calculator for the metatensor interface."""

    def __init__(self):
        super().__init__()

        # TorchScript requires to initialize all attributes in __init__
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._n_charges_channels = 0

    def forward(
        self,
        systems: Union[List[System], System],
        neighbors: Union[List[Optional[TensorBlock]], Optional[TensorBlock]] = None,
    ) -> TensorMap:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(systems, neighbors)

    def _validate_compute_parameters(
        self,
        systems: Union[List[System], System],
        neighbors: Union[List[Optional[TensorBlock]], Optional[TensorBlock]],
    ) -> Tuple[List[System], List[Optional[TensorBlock]]]:
        # check that all inputs are of the same type

        if isinstance(systems, list):
            if neighbors is not None:
                if not isinstance(neighbors, list):
                    raise TypeError(
                        "Inconsistent parameter types. `systems` is a "
                        "list, while `neighbors` is a TensorBlock. Both need "
                        "either be a list or System/TensorBlock!"
                    )
        else:
            systems = [systems]
            if neighbors is not None:
                if isinstance(neighbors, list):
                    raise TypeError(
                        "Inconsistent parameter types. `systems` is a not "
                        "a list, while `neighbors` is a list. Both need "
                        "either be a list or System/TensorBlock!"
                    )

        if not isinstance(neighbors, list):
            neighbors = [neighbors]

        # check neighbors
        if neighbors[0] is None:
            neighbors = neighbors * len(systems)

        if len(systems) != len(neighbors):
            raise ValueError(
                f"Got inconsistent numbers of systems ({len(systems)}) and "
                f"neighbors ({len(neighbors)})"
            )

        self._dtype = systems[0].positions.dtype
        self._device = systems[0].positions.device

        _components_labels = Labels(
            ["xyz"],
            torch.arange(3, dtype=torch.int32, device=self._device).unsqueeze(1),
        )
        _properties_labels = Labels(
            ["distance"], torch.zeros(1, 1, dtype=torch.int32, device=self._device)
        )

        for system, neighbors_single in zip(systems, neighbors):
            if system.positions.dtype != self._dtype:
                raise ValueError(
                    "`dtype` of all systems must be the same, got "
                    f"{system.positions.dtype} and {self._dtype}`"
                )

            if system.positions.device != self._device:
                raise ValueError(
                    "`device` of all systems must be the same, got "
                    f"{system.positions.device} and {self._device}`"
                )

            if neighbors_single is not None:
                if neighbors_single.values.dtype != self._dtype:
                    raise ValueError(
                        f"each `neighbors` must have the same type {self._dtype} "
                        "as `systems`, got at least one `neighbors` of type "
                        f"{neighbors_single.values.dtype}"
                    )

                if neighbors_single.values.device != self._device:
                    raise ValueError(
                        f"each `neighbors` must be on the same device {self._device} "
                        "as `systems`, got at least one `neighbors` with device "
                        f"{neighbors_single.values.device}"
                    )

                # Check metadata of neighbors
                samples_names = neighbors_single.samples.names
                if (
                    len(samples_names) != 5
                    or samples_names[0] != "first_atom"
                    or samples_names[1] != "second_atom"
                    or samples_names[2] != "cell_shift_a"
                    or samples_names[3] != "cell_shift_b"
                    or samples_names[4] != "cell_shift_c"
                ):
                    raise ValueError(
                        "Invalid samples for `neighbors`: the sample names must be "
                        "'first_atom', 'second_atom', 'cell_shift_a', 'cell_shift_b', "
                        "'cell_shift_c'"
                    )

                components = neighbors_single.components
                if len(components) != 1 or components[0] != _components_labels:
                    raise ValueError(
                        "Invalid components for `neighbors`: there should be a single "
                        "'xyz'=[0, 1, 2] component"
                    )

                if neighbors_single.properties != _properties_labels:
                    raise ValueError(
                        "Invalid properties for `neighbors`: there should be a single "
                        "'distance'=0 property"
                    )

        has_charges = torch.tensor(["charges" in s.known_data() for s in systems])
        if not torch.all(has_charges):
            raise ValueError("`systems` do not consistently contain `charges` data")

        # Metatensor will issue a warning because `charges` are not a default member of
        # a System object
        self._n_charges_channels = systems[0].get_data("charges").values.shape[1]

        for i_system, system in enumerate(systems):
            n_channels = system.get_data("charges").values.shape[1]
            if n_channels != self._n_charges_channels:
                raise ValueError(
                    f"number of charges-channels in system index {i_system} "
                    f"({n_channels}) is inconsistent with first system "
                    f"({self._n_charges_channels})"
                )

        return systems, neighbors

    def compute(
        self,
        systems: Union[List[System], System],
        neighbors: Union[List[Optional[TensorBlock]], Optional[TensorBlock]] = None,
    ) -> TensorMap:
        """Compute potential for all provided ``systems``.

        All ``systems`` must have the same ``dtype`` and the same ``device``. If each
        system contains a custom data field ``charges`` the potential will be calculated
        for each "charges-channel". The number of `charges-channels` must be same in all
        ``systems``. If no "explicit" charges are set the potential will be calculated
        for each "types-channels".

        :param systems: single System or list of
            :py:class:`metatensor.torch.atomisic.System` on which to run the
            calculations. The system should have ``"charges"`` using the
            :py:meth:`add_data <metatensor.torch.atomistic.System.add_data>` method.
        :param neighbors: single TensorBlock or list of a
            :py:class:`metatensor.torch.TensorBlock` containing the **half neighbor
            list**, required for periodic computations (Ewald, PME) and optional for
            direct computations. If a neighbor list is attached to a
            :py:class`metatensor.torch.atomistic.System` it can be extracted with the
            :py:meth:`get_neighborlist
            <metatensor.torch.atomistic.System.get_neighborlist>` method.

        :return: TensorMap containing the potential of all types.
        """
        systems, neighbors = self._validate_compute_parameters(systems, neighbors)
        potentials: List[torch.Tensor] = []
        samples_list: List[torch.Tensor] = []

        for i_system, (system, neighbors_single) in enumerate(zip(systems, neighbors)):
            n_atoms = len(system)
            samples = torch.zeros((n_atoms, 2), device=self._device, dtype=torch.int32)
            samples[:, 0] = i_system
            samples[:, 1] = torch.arange(
                n_atoms, device=self._device, dtype=torch.int32
            )
            samples_list.append(samples)

            charges = system.get_data("charges").values

            if torch.all(system.cell == torch.zeros([3, 3], device=system.cell.device)):
                cell = None
            else:
                cell = system.cell

            if neighbors_single is not None:
                neighbor_indices = neighbors_single.samples.view(
                    ["first_atom", "second_atom"]
                ).values
                if self._device.type == "cpu":
                    # move data to 64-bit integers, for some reason indexing with 64-bit
                    # is a lot faster than using 32-bit integers on CPU. CUDA seems fine
                    # with either types
                    neighbor_indices = neighbor_indices.to(
                        torch.int64, memory_format=torch.contiguous_format
                    )

                neighbor_shifts = neighbors_single.samples.view(
                    ["cell_shift_a", "cell_shift_b", "cell_shift_c"]
                ).values
                if self._device.type == "cpu":
                    neighbor_shifts = neighbor_shifts.to(
                        torch.int64, memory_format=torch.contiguous_format
                    )
            else:
                neighbor_indices = None
                neighbor_shifts = None

            # `_compute_single_system` is implemented only in child classes!
            potentials.append(
                self._compute_single_system(
                    positions=system.positions,
                    charges=charges,
                    cell=cell,
                    neighbor_indices=neighbor_indices,
                    neighbor_shifts=neighbor_shifts,
                )
            )

        properties_values = torch.arange(
            self._n_charges_channels, device=self._device, dtype=torch.int32
        )

        block = TensorBlock(
            values=torch.vstack(potentials),
            samples=Labels(["system", "atom"], torch.vstack(samples_list)),
            components=[],
            properties=Labels("charges_channel", properties_values.unsqueeze(1)),
        )

        keys = Labels("_", torch.zeros(1, 1, dtype=torch.int32, device=self._device))
        return TensorMap(keys=keys, blocks=[block])
