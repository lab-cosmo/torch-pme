import math
import time
from typing import Optional

import torch
import vesin.torch

from ..calculators import Calculator, EwaldCalculator, P3MCalculator, PMECalculator
from ..potentials import CoulombPotential
from .error_bounds import EwaldErrorBounds, P3MErrorBounds, PMEErrorBounds


class TunerBase:
    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        cutoff: float,
        calculator: type[Calculator],
        params: list[dict],
        exponent: int = 1,
        neighbor_indices: Optional[torch.Tensor] = None,
        neighbor_distances: Optional[torch.Tensor] = None,
    ):
        self._validate_parameters(charges, cell, positions, exponent)
        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.cutoff = cutoff
        self.exponent = calculator.exponent
        self._dtype = cell.dtype
        self._device = cell.device

        self.calculator = calculator
        self.params = params

        self._prefac = 2 * float((charges**2).sum()) / math.sqrt(len(positions))
        self.time_func = TuningTimings(
            charges,
            cell,
            positions,
            cutoff,
            neighbor_indices,
            neighbor_distances,
            4,
            2,
            True,
        )

    def tune(self, accuracy: float = 1e-3):
        raise NotImplementedError

    @staticmethod
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

        if torch.equal(
            cell.det(), torch.full([], 0, dtype=cell.dtype, device=cell.device)
        ):
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

    def estimate_smearing(
        self,
        accuracy: float,
    ) -> float:
        """Estimate the smearing based on the error formula of the real space."""
        ratio = math.sqrt(
            -2
            * math.log(
                accuracy
                / 2
                / self._prefac
                * math.sqrt(self.cutoff * float(torch.abs(self.cell.det())))
            )
        )
        smearing = self.cutoff / ratio

        return float(smearing)


class GridSearchTuner(TunerBase):
    def tune(self, accuracy: float = 1e-3):
        if self.calculator is EwaldCalculator:
            error_bounds = EwaldErrorBounds(self.charges, self.cell, self.positions)
        elif self.calculator is PMECalculator:
            error_bounds = PMEErrorBounds(self.charges, self.cell, self.positions)
        elif self.calculator is P3MCalculator:
            error_bounds = P3MErrorBounds(self.charges, self.cell, self.positions)
        else:
            raise NotImplementedError

        smearing = self.estimate_smearing(accuracy)
        param_errors = []
        param_timings = []
        for param in self.params:
            error = error_bounds(smearing=smearing, cutoff=self.cutoff, **param)
            param_errors.append(float(error))
            if error > accuracy:
                param_timings.append(float("inf"))
                continue

            param_timings.append(self._timing(smearing, param))

        return param_errors, param_timings

    def _timing(self, smearing: float, k_space_params: dict):
        calculator = self.calculator(
            potential=CoulombPotential(
                # exponent=self.exponent,  # but only exponent = 1 is supported
                smearing=smearing,
            ),
            **k_space_params,
        )

        return self.time_func(calculator)


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
