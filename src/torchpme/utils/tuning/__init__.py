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
) -> tuple[torch.tensor, torch.tensor]:
    dtype = cell.dtype
    device = cell.device

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = min_dimension / 2.0

    smearing_init = torch.tensor(
        half_cell / 5 if smearing is None else smearing,
        dtype=dtype,
        device=device,
        requires_grad=(smearing is None),
    )

    if cutoff is None:
        # solve V_SR(cutoff) == accuracy for cutoff
        def loss(cutoff):
            return (
                torch.erfc(cutoff / math.sqrt(2) / smearing_init) / cutoff - accuracy
            ) ** 2

        cutoff_init = torch.tensor(
            half_cell, dtype=dtype, device=device, requires_grad=True
        )
        _optimize_parameters(
            params=[cutoff_init],
            loss=loss,
            accuracy=accuracy,
            max_steps=1000,
            learning_rate=0.1,
        )

    cutoff_init = torch.tensor(
        float(cutoff_init) if cutoff is None else cutoff,
        dtype=dtype,
        device=device,
        requires_grad=(cutoff is None),
    )

    return smearing_init, cutoff_init


def _validate_parameters(
    sum_squared_charges: float,
    cell: torch.Tensor,
    positions: torch.Tensor,
    exponent: int,
    accuracy: float,
) -> None:
    if sum_squared_charges <= 0:
        raise ValueError(
            f"sum of squared charges must be positive, got {sum_squared_charges}"
        )

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

    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")
