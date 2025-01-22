import torch


def _validate_parameters(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    exponent: int,
) -> None:
    if exponent != 1:
        raise NotImplementedError("Only exponent = 1 is supported")

    if list(positions.shape) != [len(positions), 3]:
        raise ValueError(
            "each `positions` must be a tensor with shape [n_atoms, 3], got at least "
            f"one tensor with shape {list(positions.shape)}"
        )

    # check shape, dtype and device of cell
    dtype = positions.dtype
    if cell.dtype != dtype:
        raise ValueError(
            f"each `cell` must have the same type {dtype} as `positions`, got at least "
            "one tensor of type "
            f"{cell.dtype}"
        )

    device = positions.device
    if cell.device != device:
        raise ValueError(
            f"each `cell` must be on the same device {device} as `positions`, got at "
            "least one tensor with device "
            f"{cell.device}"
        )

    if list(cell.shape) != [3, 3]:
        raise ValueError(
            "each `cell` must be a tensor with shape [3, 3], got at least one tensor "
            f"with shape {list(cell.shape)}"
        )

    if torch.equal(cell.det(), torch.full([], 0, dtype=cell.dtype, device=cell.device)):
        raise ValueError(
            "provided `cell` has a determinant of 0 and therefore is not valid for "
            "periodic calculation"
        )

    if charges.dtype != dtype:
        raise ValueError(
            f"each `charges` must have the same type {dtype} as `positions`, got at least "
            "one tensor of type "
            f"{charges.dtype}"
        )

    if charges.device != device:
        raise ValueError(
            f"each `charges` must be on the same device {device} as `positions`, got at "
            "least one tensor with device "
            f"{charges.device}"
        )

    if charges.dim() != 2:
        raise ValueError(
            "`charges` must be a 2-dimensional tensor, got "
            f"tensor with {charges.dim()} dimension(s) and shape "
            f"{list(charges.shape)}"
        )

    if list(charges.shape) != [len(positions), charges.shape[1]]:
        raise ValueError(
            "`charges` must be a tensor with shape [n_atoms, n_channels], with "
            "`n_atoms` being the same as the variable `positions`. Got tensor with "
            f"shape {list(charges.shape)} where positions contains "
            f"{len(positions)} atoms"
        )
