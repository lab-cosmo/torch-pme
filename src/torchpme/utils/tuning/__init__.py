import time
from typing import Optional

import torch
import vesin.torch


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
        n_repeat: int = 4,
        n_warmup: int = 2,
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
