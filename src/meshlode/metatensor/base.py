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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

        Refer to :meth:`meshlode.PMEPotential.compute()` for additional details on how
        "charges-channel" and "types-channels" are computed.

        :param systems: single System or list of
            :py:class:`metatensor.torch.atomisic.System` on which to run the
            calculations.

        :return: TensorMap containing the potential of all types.
        """
        systems = self._validate_compute_parameters(systems)
        potentials: List[torch.Tensor] = []

        for system in systems:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                charges = system.get_data("charges").values

            # try to extract neighbor list from system object
            neighbor_indices = None
            neighbor_shifts = None
            for neighbor_list_options in system.known_neighbor_lists():
                if (
                    hasattr(self, "sr_cutoff")
                    and neighbor_list_options.cutoff == self.sr_cutoff
                ):
                    neighbor_list = system.get_neighbor_list(neighbor_list_options)

                    neighbor_indices = neighbor_list.samples.values[:, :2].T
                    neighbor_shifts = neighbor_list.samples.values[:, 2:].T

                    break

            potentials.append(
                self._compute_single_system(
                    positions=system.positions,
                    charges=charges,
                    cell=system.cell,
                    neighbor_indices=neighbor_indices,
                    neighbor_shifts=neighbor_shifts,
                )
            )

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
