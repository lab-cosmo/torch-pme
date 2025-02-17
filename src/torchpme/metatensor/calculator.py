import torch
from torch import profiler

try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for torchpme.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    ) from None

from .. import calculators as torch_calculators
from ..potentials.spline import Potential


class Calculator(torch.nn.Module):
    """Base calculator for the metatensor interface.

    This is just a thin wrapper around the corresponding
    generic torch :class:`torchpme.calculators.Calculator`.
    If you want to wrap a ``metatensor`` interface around another
    calculator, you can just define the class and set the static
    member ``_base_calculator`` to the corresponding
    torch calculator.

    :param potential: every calculator requires an instance of a
        :class:`Potential <torchpme.lib.Potential>` that is used
        to compute real and k-space terms for a 2-body potential.
    """

    _base_calculator: type[torch_calculators.Calculator] = torch_calculators.Calculator

    def __init__(self, potential: Potential, **kwargs):
        super().__init__()

        self._calculator = self._base_calculator(potential, **kwargs)

        # TorchScript requires to initialize all attributes in __init__
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._n_charges_channels = 0

    @staticmethod
    def _validate_compute_parameters(
        system: System,
        neighbors: TensorBlock,
    ):
        # check that all inputs are of the same type

        dtype = system.positions.dtype
        device = system.positions.device

        _components_labels = Labels(
            ["xyz"],
            torch.arange(3, dtype=torch.int32, device=device).unsqueeze(1),
        )
        _properties_labels = Labels(
            ["distance"], torch.zeros(1, 1, dtype=torch.int32, device=device)
        )

        if system.positions.dtype != dtype:
            raise ValueError(
                "`dtype` of all systems must be the same, got "
                f"{system.positions.dtype} and {dtype}`"
            )

        if system.positions.device != device:
            raise ValueError(
                "`device` of all systems must be the same, got "
                f"{system.positions.device} and {device}`"
            )

        if neighbors.values.dtype != dtype:
            raise ValueError(
                f"each `neighbors` must have the same type {dtype} "
                "as `systems`, got at least one `neighbors` of type "
                f"{neighbors.values.dtype}"
            )

        if neighbors.values.device != device:
            raise ValueError(
                f"each `neighbors` must be on the same device {device} "
                "as `systems`, got at least one `neighbors` with device "
                f"{neighbors.values.device}"
            )

        # Check metadata of neighbors
        samples_names = neighbors.samples.names
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

        components = neighbors.components
        if len(components) != 1 or components[0] != _components_labels:
            raise ValueError(
                "Invalid components for `neighbors`: there should be a single "
                "'xyz'=[0, 1, 2] component"
            )

        if neighbors.properties != _properties_labels:
            raise ValueError(
                "Invalid properties for `neighbors`: there should be a single "
                "'distance'=0 property"
            )

        if "charges" not in system.known_data():
            raise ValueError("`system` does not contain `charges` data")

        # Metatensor will issue a warning because `charges` are not a default member of
        # a System object
        n_charges_channels = system.get_data("charges").values.shape[1]

        n_channels = system.get_data("charges").values.shape[1]
        if n_channels != n_charges_channels:
            raise ValueError(
                f"number of charges-channels in system "
                f"({n_channels}) is inconsistent with first system "
                f"({n_charges_channels})"
            )

    def forward(
        self,
        system: System,
        neighbors: TensorBlock,
    ) -> TensorMap:
        """
        Compute potential for the provided ``system``.

        All ``systems`` must have the same ``dtype`` and the same ``device``. If each
        system contains a custom data field ``charges`` the potential will be calculated
        for each "charges-channel". The number of `charges-channels` must be same in all
        ``systems``. If no "explicit" charges are set the potential will be calculated
        for each "types-channels".

        :param systems: single System or list of
            :class:`metatensor.torch.atomisic.System` on which to run the
            calculations. The system should have ``"charges"`` using the
            :meth:`add_data <metatensor.torch.atomistic.System.add_data>` method.
        :param neighbors: single TensorBlock or list of a
            :class:`metatensor.torch.TensorBlock` containing the **half neighbor
            list**, required for periodic computations (Ewald, PME) and optional for
            direct computations. If a neighbor list is attached to a
            :class`metatensor.torch.atomistic.System` it can be extracted with the
            :meth:`get_neighborlist
            <metatensor.torch.atomistic.System.get_neighborlist>` method.

        :return: TensorMap containing the potential of all types.
        """

        self._validate_compute_parameters(system, neighbors)

        # In actual computations, the data type (dtype) and device (e.g. CPU, GPU) of
        # all remaining variables need to be consistent
        self._dtype = system.positions.dtype
        self._device = system.positions.device
        self._n_charges_channels = system.get_data("charges").values.shape[1]

        n_atoms = len(system)
        samples = torch.zeros((n_atoms, 2), device=self._device, dtype=torch.int32)
        samples[:, 0] = 0
        samples[:, 1] = torch.arange(n_atoms, device=self._device, dtype=torch.int32)

        neighbor_indices = neighbors.samples.view(["first_atom", "second_atom"]).values

        if self._device.type == "cpu":
            # move data to 64-bit integers, for some reason indexing with 64-bit
            # is a lot faster than using 32-bit integers on CPU. CUDA seems fine
            # with either types
            neighbor_indices = neighbor_indices.to(
                torch.int64, memory_format=torch.contiguous_format
            )

        neighbor_distances = torch.linalg.norm(neighbors.values, dim=1).squeeze(1)

        # `calculator._compute_single_system` is implemented only in child classes!
        potential = self._calculator.forward(
            charges=system.get_data("charges").values,
            cell=system.cell,
            positions=system.positions,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )

        with profiler.record_function("wrap metatensor"):
            properties_values = torch.arange(
                self._n_charges_channels, device=self._device, dtype=torch.int32
            )

            block = TensorBlock(
                values=potential,
                samples=Labels(["system", "atom"], samples),
                components=[],
                properties=Labels("charges_channel", properties_values.unsqueeze(1)),
            )

            keys = Labels(
                "_", torch.zeros(1, 1, dtype=torch.int32, device=self._device)
            )
            return TensorMap(keys=keys, blocks=[block])
