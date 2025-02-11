from typing import Union

import torch


def _validate_parameters(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    smearing: Union[float, None],
) -> None:
    dtype = positions.dtype
    device = positions.device

    # check shape, dtype and device of positions
    num_atoms = len(positions)
    if list(positions.shape) != [len(positions), 3]:
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

    if smearing is not None and torch.equal(
        cell.det(), torch.tensor(0.0, dtype=cell.dtype, device=cell.device)
    ):
        raise ValueError(
            "provided `cell` has a determinant of 0 and therefore is not valid for "
            "periodic calculation"
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
