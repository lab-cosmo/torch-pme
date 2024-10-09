import torch

def _validate_forward_parameters(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
) -> None:
    device = positions.device
    dtype = positions.dtype

    # check shape, dtype and device of positions
    num_atoms = len(positions)
    if list(positions.shape) != [len(positions), 3]:
        raise ValueError(
            "each `positions` must be a tensor with shape [n_atoms, 3], got at "
            f"least one tensor with shape {list(positions.shape)}"
        )

    # check shape, dtype and device of cell
    if list(cell.shape) != [3, 3]:
        raise ValueError(
            "each `cell` must be a tensor with shape [3, 3], got at least "
            f"one tensor with shape {list(cell.shape)}"
        )

    if cell.dtype != dtype:
        raise ValueError(
            f"each `cell` must have the same type {dtype} as "
            "`positions`, got at least one tensor of type "
            f"{cell.dtype}"
        )

    if cell.device != device:
        raise ValueError(
            f"each `cell` must be on the same device {device} as "
            "`positions`, got at least one tensor with device "
            f"{cell.device}"
        )

    # check shape, dtype & device of `charges`
    if charges.dim() != 2:
        raise ValueError(
            "each `charges` needs to be a 2-dimensional tensor, got at least "
            f"one tensor with {charges.dim()} dimension(s) and shape "
            f"{list(charges.shape)}"
        )

    if list(charges.shape) != [num_atoms, charges.shape[1]]:
        raise ValueError(
            "each `charges` must be a tensor with shape [n_atoms, n_channels], "
            "with `n_atoms` being the same as the variable `positions`. Got at "
            f"least one tensor with shape {list(charges.shape)} where "
            f"positions contains {len(positions)} atoms"
        )

    if charges.dtype != dtype:
        raise ValueError(
            f"each `charges` must have the same type {dtype} as "
            "`positions`, got at least one tensor of type "
            f"{charges.dtype}"
        )

    if charges.device != device:
        raise ValueError(
            f"each `charges` must be on the same device {device} as "
            f"`positions`, got at least one tensor with device "
            f"{charges.device}"
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
            f"each `neighbor_indices` must be on the same device "
            f"{device} as `positions`, got at least one tensor with "
            f"device {neighbor_indices.device}"
        )

    if neighbor_distances.shape != neighbor_indices[:, 0].shape:
        raise ValueError(
            "`neighbor_indices` and `neighbor_distances` need to have shapes "
            "[num_neighbors, 2] and [num_neighbors]. For at least one "
            f"structure, got {list(neighbor_indices.shape)} and "
            f"{list(neighbor_distances.shape)}, "
            "which is inconsistent"
        )

    if neighbor_distances.device != device:
        raise ValueError(
            f"each `neighbor_distances` must be on the same device "
            f"{device} as `positions`, got at least one tensor with "
            f"device {neighbor_distances.device}"
        )