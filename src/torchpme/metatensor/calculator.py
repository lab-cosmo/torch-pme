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
from ..lib import batching as _batching


def _validate_system_parameters(system: System, neighbors: TensorBlock) -> None:
    """Validate one system + neighbors pair (shared by serial and batched paths)."""
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

    # compare the names and values directly instead of building throwaway `Labels` to
    # compare against: this runs on every forward call, and constructing plus comparing
    # the two `Labels` objects costs ~34 us per system against ~10 us for the checks
    # below (the metadata they accept is the same)
    components = neighbors.components
    if (
        len(components) != 1
        or components[0].names != ["xyz"]
        or not torch.equal(
            components[0].values.flatten(),
            torch.arange(3, dtype=torch.int32, device=device),
        )
    ):
        raise ValueError(
            "Invalid components for `neighbors`: there should be a single "
            "'xyz'=[0, 1, 2] component"
        )

    properties = neighbors.properties
    if (
        properties.names != ["distance"]
        or len(properties) != 1
        or int(properties.values[0, 0]) != 0
    ):
        raise ValueError(
            "Invalid properties for `neighbors`: there should be a single "
            "'distance'=0 property"
        )

    if "charge" not in system.known_data():
        raise ValueError("`system` does not contain `charge` data")

    charge_tensor = system.get_data("charge")
    if len(charge_tensor) != 1:
        raise ValueError(
            f"Charge tensor have exactlty one block but has {len(charge_tensor)} blocks"
        )

    n_charge_components = len(charge_tensor.block().components)
    if n_charge_components > 0:
        raise ValueError(
            "TensorBlock containg the charges should not have components; "
            f"found {n_charge_components}"
        )


def _neighbor_indices(neighbors: TensorBlock) -> torch.Tensor:
    """Extract the ``[n_pairs, 2]`` atom-index tensor from a neighbors block."""
    neighbor_indices = torch.stack(
        [
            neighbors.samples.column("first_atom"),
            neighbors.samples.column("second_atom"),
        ],
        dim=1,
    )
    if neighbor_indices.device.type == "cpu":
        # move to 64-bit integers, for some reason indexing 64-bit is a lot faster
        # than using 32-bit integers on CPU. CUDA seems fine with either types
        neighbor_indices = neighbor_indices.to(
            torch.int64, memory_format=torch.contiguous_format
        )
    return neighbor_indices


def prepare_tiled_batch(
    systems: list[System],
    neighbors: list[TensorBlock],
    num_k: int,
    block_atoms: int = 32,
    block_kvecs: int = 128,
    halfspace: bool = True,
    k_pad_fraction: float = 0.1,
    smearing_factor: float = 2.0,
    smearing: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Build the static tiling data for :meth:`Calculator.forward_batched` from a list of
    systems.

    This is the ``metatensor`` front-end of :func:`torchpme.lib.prepare_tiled_batch`;
    see there for the meaning of the parameters. It runs on the host, once per batch
    (the tiling depends only on the cells, periodicities, atom counts and pair counts —
    and, for 2D slabs, on the extent of the atoms along the vacuum axis). Like the
    calculator's ``forward_batched``, it is TorchScript-compatible.

    Each *periodic* system's neighbor list must have been built at that system's own
    cutoff derived from ``num_k`` (see :func:`torchpme.lib.ewald_params_from_num_k`);
    for non-periodic systems the list defines the pairs over which the bare potential
    is summed.

    :param systems: systems with attached ``"charge"`` data, as in
        :meth:`Calculator.forward`
    :param neighbors: one neighbor :class:`TensorBlock
        <metatensor.torch.TensorBlock>` per system, same conventions as in
        :meth:`Calculator.forward`
    :param num_k: target half-space k-vector count per system
    :param block_atoms: atom-tile size of the reciprocal-space kernel
    :param block_kvecs: k-tile size of the reciprocal-space kernel
    :param halfspace: keep only one of each ±k pair (with degeneracy factor 2)
    :param k_pad_fraction: padding window of the common k-count
    :param smearing_factor: ratio of the smearing to the reciprocal resolution; must
        match the value used to compute the neighbor-list cutoffs
    :param smearing: optional override of the derived per-system smearing, as a 0-dim
        tensor or one of shape ``(n_systems,)``
    :return: the tiling dictionary consumed by :meth:`Calculator.forward_batched`, on
        the device of the systems
    """
    if len(systems) != len(neighbors):
        raise ValueError(
            f"Got {len(systems)} systems but {len(neighbors)} neighbor blocks"
        )
    # the full metadata validation happens in `forward_batched`, which is what consumes
    # the neighbor blocks; running it here as well would double its cost per batch. The
    # collation only needs the charges, so check just those.
    for system in systems:
        if "charge" not in system.known_data():
            raise ValueError("`system` does not contain `charge` data")

    _, tiling = _batching.prepare_tiled_batch(
        positions=[system.positions for system in systems],
        charges=[system.get_data("charge").block().values for system in systems],
        cells=[system.cell for system in systems],
        periodic=[system.pbc for system in systems],
        neighbor_indices=[_neighbor_indices(block) for block in neighbors],
        neighbor_distances=[
            torch.linalg.norm(block.values, dim=1).squeeze(-1).detach()
            for block in neighbors
        ],
        num_k=num_k,
        block_atoms=block_atoms,
        block_kvecs=block_kvecs,
        halfspace=halfspace,
        k_pad_fraction=k_pad_fraction,
        smearing_factor=smearing_factor,
        smearing=smearing,
    )
    return tiling


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
        _validate_system_parameters(system, neighbors)

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
        charges = system.get_data("charge").block().values

        n_atoms = len(system)
        samples = torch.zeros((n_atoms, 2), device=device, dtype=torch.int32)
        samples[:, 0] = 0
        samples[:, 1] = torch.arange(n_atoms, device=device, dtype=torch.int32)

        neighbor_indices = _neighbor_indices(neighbors)
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

    @torch.jit.export
    def forward_batched(
        self,
        systems: list[System],
        neighbors: list[TensorBlock],
        tiling: dict[str, torch.Tensor],
    ) -> TensorMap:
        """
        Compute the potential for a tiled batch of systems in a single call.

        This is the ``metatensor`` front-end of the batched (tiled) evaluation; the
        underlying computation and the batch layout are documented in
        :meth:`torchpme.EwaldCalculator.forward_batched`, which is the only calculator
        implementing it. The static ``tiling`` data must be built beforehand — once
        per batch — with :func:`torchpme.metatensor.prepare_tiled_batch` from the
        *same* systems and neighbor lists, in the same order.

        Each system must carry ``"charge"`` data with the same number of channels, and
        each neighbor block follows the same conventions as in :meth:`forward`. The
        pair distances are recomputed from the neighbor distance vectors inside the
        computational graph, so forces flow to the positions used to compute those
        vectors.

        :param systems: systems to evaluate, in the tiling's order
        :param neighbors: one neighbor list per system, in the tiling's order
        :param tiling: static tiling data from
            :func:`torchpme.metatensor.prepare_tiled_batch`
        :return: :class:`metatensor.torch.TensorMap` with one block whose samples are
            ``("system", "atom")`` (system index in the batch, atom index within the
            system) and whose properties are the ``charges_channel``
        """
        if len(systems) != len(neighbors):
            raise ValueError(
                f"Got {len(systems)} systems but {len(neighbors)} neighbor blocks"
            )
        if len(systems) == 0:
            raise ValueError("Cannot evaluate an empty list of systems")

        for i in range(len(systems)):
            _validate_system_parameters(systems[i], neighbors[i])

        device = systems[0].positions.device

        # concatenate the per-system data in list order; pairs follow the collate's
        # order: periodic systems' (screened) pairs first, non-periodic (bare) after
        positions_list: list[torch.Tensor] = []
        charges_list: list[torch.Tensor] = []
        cell_list: list[torch.Tensor] = []
        periodic_list: list[torch.Tensor] = []
        system_index_list: list[torch.Tensor] = []
        samples_list: list[torch.Tensor] = []
        screened_indices: list[torch.Tensor] = []
        screened_distances: list[torch.Tensor] = []
        bare_indices: list[torch.Tensor] = []
        bare_distances: list[torch.Tensor] = []

        n_channels = systems[0].get_data("charge").block().values.shape[1]
        atom_offset = 0
        for index in range(len(systems)):
            system = systems[index]
            block = neighbors[index]
            charges = system.get_data("charge").block().values
            if charges.shape[1] != n_channels:
                raise ValueError(
                    "All systems must have the same number of charge channels, got "
                    f"{charges.shape[1]} and {n_channels}"
                )
            n_atoms = len(system)
            positions_list.append(system.positions)
            charges_list.append(charges)
            cell_list.append(system.cell)
            periodic_list.append(system.pbc)
            system_index_list.append(
                torch.full((n_atoms,), index, dtype=torch.long, device=device)
            )
            samples = torch.zeros((n_atoms, 2), dtype=torch.int32, device=device)
            samples[:, 0] = index
            samples[:, 1] = torch.arange(n_atoms, dtype=torch.int32, device=device)
            samples_list.append(samples)

            indices = _neighbor_indices(block) + atom_offset
            distances = torch.linalg.norm(block.values, dim=1).squeeze(1)
            if bool(system.pbc.any()):
                screened_indices.append(indices)
                screened_distances.append(distances)
            else:
                bare_indices.append(indices)
                bare_distances.append(distances)
            atom_offset += n_atoms

        positions = torch.cat(positions_list)
        empty_indices = torch.empty(0, 2, dtype=torch.long, device=device)
        empty_distances = torch.empty(0, dtype=positions.dtype, device=device)
        neighbor_indices = torch.cat(screened_indices + bare_indices + [empty_indices])
        neighbor_distances = torch.cat(
            screened_distances + bare_distances + [empty_distances]
        )

        potential = self._calculator.forward_batched(
            charges=torch.cat(charges_list),
            cell=torch.stack(cell_list),
            positions=positions,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            system_index=torch.cat(system_index_list),
            periodic=torch.stack(periodic_list),
            tiling=tiling,
        )

        properties_values = torch.arange(n_channels, dtype=torch.int32, device=device)
        block_out = TensorBlock(
            values=potential,
            samples=Labels(["system", "atom"], torch.cat(samples_list)),
            components=[],
            properties=Labels("charges_channel", properties_values.unsqueeze(1)),
        )
        keys = Labels("_", torch.zeros(1, 1, dtype=torch.int32, device=device))
        return TensorMap(keys=keys, blocks=[block_out])
