import torch


def _validate_batched_parameters(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    system_index: torch.Tensor,
    periodic: torch.Tensor,
    tiling: dict[str, torch.Tensor],
) -> None:
    dtype = positions.dtype
    device = positions.device

    num_atoms = positions.shape[0]
    if positions.dim() != 2 or positions.shape[1] != 3:
        raise ValueError(
            "`positions` must be a tensor with shape [n_atoms, 3], got tensor "
            f"with shape {list(positions.shape)}"
        )

    if cell.dim() != 3 or cell.shape[1] != 3 or cell.shape[2] != 3:
        raise ValueError(
            "`cell` must be a tensor with shape [n_systems, 3, 3], got tensor with "
            f"shape {list(cell.shape)}"
        )
    if cell.dtype != dtype or cell.device != device:
        raise ValueError(
            "`cell` must have the same dtype and device as `positions`, got "
            f"{cell.dtype} on {cell.device}"
        )
    num_systems = cell.shape[0]

    if (
        periodic.dim() != 2
        or periodic.shape[0] != num_systems
        or periodic.shape[1] != 3
    ):
        raise ValueError(
            "`periodic` must be a tensor with shape [n_systems, 3], got tensor with "
            f"shape {list(periodic.shape)}"
        )
    if periodic.dtype != torch.bool:
        raise TypeError(f"type of `periodic` ({periodic.dtype}) must be torch.bool")
    if bool((periodic.sum(dim=-1) == 1).any()):
        raise ValueError("1D-periodic systems are not supported")

    if charges.dim() != 2 or charges.shape[0] != num_atoms:
        raise ValueError(
            "`charges` must be a tensor with shape [n_atoms, n_channels], got tensor "
            f"with shape {list(charges.shape)} for {num_atoms} atoms"
        )
    if charges.dtype != dtype or charges.device != device:
        raise ValueError(
            "`charges` must have the same dtype and device as `positions`, got "
            f"{charges.dtype} on {charges.device}"
        )

    if system_index.dim() != 1 or system_index.shape[0] != num_atoms:
        raise ValueError(
            "`system_index` must be a tensor with shape [n_atoms], got tensor with "
            f"shape {list(system_index.shape)} for {num_atoms} atoms"
        )

    if neighbor_indices.dim() != 2 or neighbor_indices.shape[1] != 2:
        raise ValueError(
            "neighbor_indices is expected to have shape [num_neighbors, 2], but got "
            f"{list(neighbor_indices.shape)}"
        )
    if neighbor_distances.shape != neighbor_indices[:, 0].shape:
        raise ValueError(
            "`neighbor_indices` and `neighbor_distances` need to have shapes "
            "[num_neighbors, 2] and [num_neighbors], but got "
            f"{list(neighbor_indices.shape)} and {list(neighbor_distances.shape)}"
        )
    if neighbor_distances.dtype != dtype or neighbor_distances.device != device:
        raise ValueError(
            "`neighbor_distances` must have the same dtype and device as `positions`, "
            f"got {neighbor_distances.dtype} on {neighbor_distances.device}"
        )

    required_keys = [
        "k_int",
        "sigma",
        "sigma_pair",
        "pbc_system",
        "periodic_rows",
        "effective_cell_static",
        "b_col",
        "kt_col",
        "atom_gather",
        "gather_mask",
        "flat_to_atom",
        "flat_mask",
        "pbc_atom",
        "block_atoms",
        "block_kvecs",
        "n_kvec_tiles",
        "n_screened_pairs",
        "g_factor",
    ]
    for key in required_keys:
        if key not in tiling:
            raise ValueError(
                f"`tiling` is missing the required key '{key}'; build it with "
                "torchpme.lib.prepare_tiled_batch"
            )


def _validate_parameters(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    periodic: torch.Tensor | None = None,
    pair_mask: torch.Tensor | None = None,
    node_mask: torch.Tensor | None = None,
    kvectors: torch.Tensor | None = None,
) -> None:
    dtype = positions.dtype
    device = positions.device

    # check shape, dtype and device of positions
    num_atoms = positions.shape[-2]
    if list(positions.shape) != [num_atoms, 3]:
        raise ValueError(
            "`positions` must be a tensor with shape [n_atoms, 3], got tensor "
            f"with shape {list(positions.shape)}"
        )

    # check shape, dtype and device of cell
    if list(cell.shape) != [3, 3]:
        raise ValueError(
            "`cell` must be a tensor with shape [3, 3], got tensor with shape "
            f"{list(cell.shape)}"
        )

    if cell.dtype != dtype:
        raise TypeError(
            f"type of `cell` ({cell.dtype}) must be same as that of the `positions` class ({dtype})"
        )

    if cell.device != device:
        raise ValueError(
            f"device of `cell` ({cell.device}) must be same as that of the `positions` class ({device})"
        )

    # check shape, dtype & device of `charges`
    if charges.dim() != 2:
        raise ValueError(
            "`charges` must be a 2-dimensional tensor, got "
            f"tensor with {charges.dim()} dimension(s) and shape "
            f"{list(charges.shape)}"
        )

    if list(charges.shape) != [num_atoms, charges.shape[1]]:
        raise ValueError(
            "`charges` must be a tensor with shape [n_atoms, n_channels], with "
            "`n_atoms` being the same as the variable `positions`. Got tensor with "
            f"shape {list(charges.shape)} where positions contains "
            f"{len(positions)} atoms"
        )

    if charges.dtype != dtype:
        raise TypeError(
            f"type of `charges` ({charges.dtype}) must be same as that of the `positions` class ({dtype})"
        )

    if charges.device != device:
        raise ValueError(
            f"device of `charges` ({charges.device}) must be same as that of the `positions` class "
            f"({device})"
        )

    # check shape, dtype & device of `neighbor_indices` and `neighbor_distances`
    if neighbor_indices.shape[1] != 2:
        raise ValueError(
            "neighbor_indices is expected to have shape [num_neighbors, 2]"
            f", but got {list(neighbor_indices.shape)} for one "
            "structure"
        )

    if neighbor_indices.device != device:
        raise ValueError(
            f"device of `neighbor_indices` ({neighbor_indices.device}) must be "
            f"same as that of the `positions` class ({device})"
        )

    if neighbor_distances.shape != neighbor_indices[:, 0].shape:
        raise ValueError(
            "`neighbor_indices` and `neighbor_distances` need to have shapes "
            "[num_neighbors, 2] and [num_neighbors], but got "
            f"{list(neighbor_indices.shape)} and {list(neighbor_distances.shape)}"
        )

    if neighbor_distances.device != device:
        raise ValueError(
            f"device of `neighbor_distances` ({neighbor_distances.device}) must be "
            f"same as that of the `positions` class ({device})"
        )

    if neighbor_distances.dtype != dtype:
        raise TypeError(
            f"type of `neighbor_distances` ({neighbor_distances.dtype}) must be same "
            f"as that of the `positions` class ({dtype})"
        )

    if periodic is not None:
        if periodic.shape != (3,):
            raise ValueError(
                "`periodic` must be a tensor of shape (3,), got "
                f"tensor with shape {list(periodic.shape)}"
            )

        if periodic.device != device:
            raise ValueError(
                f"device of `periodic` ({periodic.device}) must be same as that of "
                f"the `positions` class ({device})"
            )

        if periodic.dtype != torch.bool:
            raise TypeError(f"type of `periodic` ({periodic.dtype}) must be torch.bool")

    if pair_mask is not None:
        if pair_mask.shape != neighbor_indices[:, 0].shape:
            raise ValueError(
                "`pair_mask` must have the same shape as the number of neighbors, "
                f"got tensor with shape {list(pair_mask.shape)} while the number of "
                f"neighbors is {neighbor_indices.shape[0]}"
            )

        if pair_mask.device != device:
            raise ValueError(
                f"device of `pair_mask` ({pair_mask.device}) must be same as that "
                f"of the `positions` class ({device})"
            )

        if pair_mask.dtype != torch.bool:
            raise TypeError(
                f"type of `pair_mask` ({pair_mask.dtype}) must be torch.bool"
            )

    if node_mask is not None:
        if node_mask.shape != (num_atoms,):
            raise ValueError(
                "`node_mask` must have shape [n_atoms], got tensor with shape "
                f"{list(node_mask.shape)} where n_atoms is {num_atoms}"
            )

        if node_mask.device != device:
            raise ValueError(
                f"device of `node_mask` ({node_mask.device}) must be same as that "
                f"of the `positions` class ({device})"
            )

        if node_mask.dtype != torch.bool:
            raise TypeError(
                f"type of `node_mask` ({node_mask.dtype}) must be torch.bool"
            )

    if kvectors is not None:
        if kvectors.shape[1] != 3:
            raise ValueError(
                "`kvectors` must be a tensor of shape [n_kvecs, 3], got "
                f"tensor with shape {list(kvectors.shape)}"
            )

        if kvectors.device != device:
            raise ValueError(
                f"device of `kvectors` ({kvectors.device}) must be same as that of "
                f"the `positions` class ({device})"
            )

        if kvectors.dtype != dtype:
            raise TypeError(
                f"type of `kvectors` ({kvectors.dtype}) must be same as that of the "
                f"`positions` class ({dtype})"
            )
