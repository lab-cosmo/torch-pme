import math
from typing import Optional

import time
import torch
import vesin.torch


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


class TuningErrorBounds(torch.nn.Module):
    """Base class for error bounds."""

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ):
        super().__init__()
        self._charges = charges
        self._cell = cell
        self._positions = positions

    def forward(self, *args, **kwargs):
        return self.error(*args, **kwargs)
    

class TuningTimings(torch.nn.Module):
    """Base class for error bounds."""

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        cutoff: float,
        neighbor_indices: Optional[torch.Tensor] = None,
        neighbor_distances: Optional[torch.Tensor] = None,
        n_repeat: Optional[int] = 4,
        n_warmup: Optional[int] = 2,
        run_backward: Optional[bool] = True,
    ):
        super().__init__()
        self._charges = charges
        self._cell = cell
        self._positions = positions
        self._dtype = charges.dtype
        self._device = charges.device
        self._n_repeat = n_repeat
        self._n_warmup = n_warmup
        self._run_backward = run_backward

        if neighbor_indices is None and neighbor_distances is None:
            nl = vesin.torch.NeighborList(cutoff=cutoff, full_list=False)
            i, j, neighbor_distances = nl.compute(
                points=self._positions.to(dtype=torch.float64, device="cpu"),
                box=self._cell.to(dtype=torch.float64, device="cpu"),
                periodic=True,
                quantities="ijd",
            )
            neighbor_indices = torch.stack([i, j], dim=1)
        elif neighbor_indices is None or neighbor_distances is None:
            raise ValueError(
                "If neighbor_indices or neighbor_distances are None, "
                "both must be None."
            )
        self._neighbor_indices = neighbor_indices.to(device=self._device)
        self._neighbor_distances = neighbor_distances.to(
            dtype=self._dtype, device=self._device
        )

    def forward(self, calculator: torch.nn.Module):
        """
        Estimate the execution time of a given calculator for the structure
        to be used as benchmark.
        """

        for _ in range(self._n_warmup):
            result = calculator.forward(
                positions=self._positions,
                charges=self._charges,
                cell=self._cell,
                neighbor_indices=self._neighbor_indices,
                neighbor_distances=self._neighbor_distances,
            )

        # measure time
        execution_time = 0.0

        for _ in range(self._n_repeat):
            positions = self._positions.clone()
            cell = self._cell.clone()
            charges = self._charges.clone()
            # nb - this won't compute gradiens involving the distances
            if self._run_backward:
                positions.requires_grad_(True)
                cell.requires_grad_(True)
                charges.requires_grad_(True)
            execution_time -= time.time()            
            result = calculator.forward(
                positions=positions,
                charges=charges,
                cell=cell,
                neighbor_indices=self._neighbor_indices,
                neighbor_distances=self._neighbor_distances,
                )
            value = result.sum()
            if self._run_backward:
                value.backward(retain_graph=True)

            if self._device is torch.device("cuda"):
                torch.cuda.synchronize()
            execution_time += time.time()

        return execution_time / self._n_repeat