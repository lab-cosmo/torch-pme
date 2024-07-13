import warnings
from typing import List, Union

import torch


try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for meshlode.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    )


class CalculatorBaseMetatensor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._n_charges_channels = 0

    def forward(self, systems: Union[List[System], System]) -> TensorMap:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(systems)

    def _validate_compute_parameters(
        self, systems: Union[List[System], System]
    ) -> List[System]:
        # Make sure that the compute function also works if only a single frame is
        # provided as input (for convenience of users testing out the code)
        if not isinstance(systems, list):
            systems = [systems]

        self._device = systems[0].positions.device
        for system in systems:
            if system.dtype != systems[0].dtype:
                raise ValueError(
                    "`dtype` of all systems must be the same, got "
                    f"{system.dtype} and {systems[0].dtype}`"
                )

            if system.device != self._device:
                raise ValueError(
                    "`device` of all systems must be the same, got "
                    f"{system.device} and {systems[0].device}`"
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

        return systems

    def compute(self, systems: Union[List[System], System]) -> TensorMap:
        """Compute potential for all provided ``systems``.

        All ``systems`` must have the same ``dtype`` and the same ``device``. If each
        system contains a custom data field ``charges`` the potential will be calculated
        for each "charges-channel". The number of `charges-channels` must be same in all
        ``systems``. If no "explicit" charges are set the potential will be calculated
        for each "types-channels".

        :param systems: single System or list of
            :py:class:`metatensor.torch.atomisic.System` on which to run the
            calculations. The system should have ``"charges"`` using the
            :py:meth:`add_data <metatensor.torch.atomistic.System.add_data>` method. If
            periodic computations (Ewald, PME) are performed additionally a **full
            neighbor list** should be attached using the :py:meth:`add_neighbor_list
            <metatensor.torch.atomistic.System.add_neighbor_list>` method. If a
            ``system`` has *multiple* neighbor lists the first full list will taken into
            account for the computation.

        :return: TensorMap containing the potential of all types.
        """
        systems = self._validate_compute_parameters(systems)
        potentials: List[torch.Tensor] = []

        for system in systems:
            charges = system.get_data("charges").values
            all_neighbor_lists = system.known_neighbor_lists()
            if all_neighbor_lists:
                # try to extract neighbor list from system object
                has_full_neighbor_list = False
                first_full_neighbor_list = all_neighbor_lists[0]
                for neighbor_list_options in all_neighbor_lists:
                    if neighbor_list_options.full_list:
                        has_full_neighbor_list = True
                        first_full_neighbor_list = neighbor_list_options
                        break

                if not has_full_neighbor_list:
                    raise ValueError(
                        f"Found {len(all_neighbor_lists)} neighbor list(s) but no full "
                        "list, which is required."
                    )

                if len(system.known_neighbor_lists()) > 1:
                    warnings.warn(
                        "Multiple neighbor lists found "
                        f"({len(all_neighbor_lists)}). Using the full first one "
                        f"with `cutoff={first_full_neighbor_list.cutoff}`.",
                        stacklevel=2,
                    )

                neighbor_list = system.get_neighbor_list(first_full_neighbor_list)
                neighbor_indices = neighbor_list.samples.values[:, :2].T
                neighbor_shifts = neighbor_list.samples.values[:, 2:]
            else:
                neighbor_indices = None
                neighbor_shifts = None

            potentials.append(
                self._compute_single_system(
                    positions=system.positions,
                    charges=charges,
                    cell=system.cell,
                    neighbor_indices=neighbor_indices,
                    neighbor_shifts=neighbor_shifts,
                )
            )
        system = systems[-1]
        values_samples: List[List[int]] = []
        for i_system in range(len(systems)):
            for i_atom in range(len(system)):
                values_samples.append([i_system, i_atom])

        samples_vals_tensor = torch.tensor(values_samples, device=self._device)

        block = TensorBlock(
            values=torch.vstack(potentials),
            samples=Labels(["system", "atom"], samples_vals_tensor),
            components=[],
            properties=Labels.range("charges_channel", self._n_charges_channels),
        )

        return TensorMap(keys=Labels.single(), blocks=[block])
