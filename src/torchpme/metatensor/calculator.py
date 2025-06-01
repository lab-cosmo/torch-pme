import torch

try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
except ImportError:
    raise ImportError(
        "metatensor.torch is required for torchpme.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    ) from None

try:
    from metatomic.torch import System
except ImportError:
    raise ImportError(
        "metatomic is required for torchpme.metatensor but is not installed. "
        "Try installing it with:\npip install metatomic"
    ) from None

from .. import calculators as torch_calculators


class Calculator(torch.nn.Module):
    """
    Base calculator for the metatensor interface.

    This is just a thin wrapper around the corresponding generic torch
    :class:`torchpme.calculators.Calculator`. If you want to wrap a ``metatensor``
    interface around another calculator, you can just define the class and set the
    static member ``_base_calculator`` to the corresponding torch calculator.
    """

    _base_calculator: type[torch_calculators.Calculator] = torch_calculators.Calculator

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._calculator = self._base_calculator(*args, **kwargs)

    @staticmethod
    def _validate_compute_parameters(system: System, neighbors: TensorBlock) -> None:
        dtype = system.positions.dtype
        device = system.positions.device

        if neighbors.values.dtype != dtype:
            raise ValueError(
                f"dtype of `neighbors` ({neighbors.values.dtype}) must be the same "
                f"as `system` ({dtype})"
            )

        if neighbors.values.device != device:
            raise ValueError(
                f"device of `neighbors` ({neighbors.values.device}) must be the same "
                f"as `system` ({device})"
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

        components_labels = Labels(
            ["xyz"],
            torch.arange(3, dtype=torch.int32, device=device).unsqueeze(1),
        )
        components = neighbors.components
        if len(components) != 1 or components[0] != components_labels:
            raise ValueError(
                "Invalid components for `neighbors`: there should be a single "
                "'xyz'=[0, 1, 2] component"
            )

        properties_labels = Labels(
            ["distance"], torch.zeros(1, 1, dtype=torch.int32, device=device)
        )
        if neighbors.properties != properties_labels:
            raise ValueError(
                "Invalid properties for `neighbors`: there should be a single "
                "'distance'=0 property"
            )

        if "charges" not in system.known_data():
            raise ValueError("`system` does not contain `charges` data")

        charge_tensor = system.get_data("charges")
        if len(charge_tensor) != 1:
            raise ValueError(
                f"Charge tensor have exactlty one block but has {len(charge_tensor)} "
                "blocks"
            )

        n_charge_components = len(charge_tensor.block().components)
        if n_charge_components > 0:
            raise ValueError(
                "TensorBlock containg the charges should not have components; "
                f"found {n_charge_components}"
            )

    def forward(self, system: System, neighbors: TensorBlock) -> TensorMap:
        """
        Compute the potential "energy".

        The ``system`` must contain a custom data field ``charges``. The potential will
        be calculated for each ``"charges_channel"``, which will also be the properties
        name of the returned :class:`metatensor.torch.TensorMap`.

        :param system: System to run the calculations. The system must have attached
            ``"charges"`` using the :meth:`add_data
            <metatomic.torch.System.add_data>` method.
        :param neighbors: The neighbor list. If a neighbor list is attached to a
            :class:`metatomic.torch.System` it can be extracted with the
            :meth:`get_neighborlist
            <metatomic.torch.System.get_neighborlist>` method using a
            :class:`NeighborListOptions
            <metatomic.torch.NeighborListOptions>`. Note to use the same
            ``full_list`` option for these options as provided for
            ``full_neighbor_list`` in the constructor.

            .. note::

                Although ``neighbors`` can be attached to the ``system``, they are
                required to be passed explicitly here. While it's possible to design the
                class to automatically extract the neighbor list by accepting a
                :class:`NeighborListOptions
                <metatomic.torch.NeighborListOptions>` directly in the
                constructor, we chose explicit passing for consistency with the torch
                interface.

        :return: :class:`metatensor.torch.TensorMap` containing the potential
        """
        self._validate_compute_parameters(system, neighbors)

        device = system.positions.device
        charges = system.get_data("charges").block().values

        n_atoms = len(system)
        samples = torch.zeros((n_atoms, 2), device=device, dtype=torch.int32)
        samples[:, 0] = 0
        samples[:, 1] = torch.arange(n_atoms, device=device, dtype=torch.int32)

        neighbor_indices = neighbors.samples.view(["first_atom", "second_atom"]).values

        if device.type == "cpu":
            # move to 64-bit integers, for some reason indexing 64-bit is a lot faster
            # than using 32-bit integers on CPU. CUDA seems fine with either types
            neighbor_indices = neighbor_indices.to(
                torch.int64, memory_format=torch.contiguous_format
            )

        neighbor_distances = torch.linalg.norm(neighbors.values, dim=1).squeeze(1)

        potential = self._calculator.forward(
            charges=charges,
            cell=system.cell,
            positions=system.positions,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )

        properties_values = torch.arange(
            charges.shape[1], device=device, dtype=torch.int32
        )

        block = TensorBlock(
            values=potential,
            samples=Labels(["system", "atom"], samples),
            components=[],
            properties=Labels("charges_channel", properties_values.unsqueeze(1)),
        )

        keys = Labels("_", torch.zeros(1, 1, dtype=torch.int32, device=device))
        return TensorMap(keys=keys, blocks=[block])
