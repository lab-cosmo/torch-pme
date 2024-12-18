import math
import warnings
from typing import Callable, Optional

import torch


def _optimize_parameters(
    params: list[torch.Tensor],
    loss: Callable,
    max_steps: int,
    accuracy: float,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    for _ in range(max_steps):
        loss_value = loss(*params)
        if torch.isnan(loss_value) or torch.isinf(loss_value):
            raise ValueError(
                "The value of the estimated error is now nan, consider using a "
                "smaller learning rate."
            )
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss_value <= accuracy:
            break

    if loss_value > accuracy:
        warnings.warn(
            "The searching for the parameters is ended, but the error is "
            f"{float(loss_value):.3e}, larger than the given accuracy {accuracy}. "
            "Consider increase max_step and",
            stacklevel=2,
        )


def _estimate_smearing_cutoff(
    cell: torch.Tensor,
    smearing: Optional[float],
    cutoff: Optional[float],
    accuracy: float,
    prefac: float,
) -> tuple[float, float]:
    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = min_dimension / 2.0
    cutoff_init = min(5.0, half_cell) if cutoff is None else cutoff
    ratio = math.sqrt(
        -2
        * math.log(
            accuracy
            / 2
            / prefac
            * math.sqrt(cutoff_init * float(torch.abs(cell.det())))
        )
    )
    smearing_init = cutoff_init / ratio if smearing is None else smearing

    return float(smearing_init), float(cutoff_init)


def _validate_parameters(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    exponent: int,
    accuracy: float,
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

    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")
