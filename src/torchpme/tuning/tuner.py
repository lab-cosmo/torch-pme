import math
import time
from typing import Optional

import torch
import vesin.torch

from ..calculators import Calculator
from ..potentials import CoulombPotential
from ..utils import _validate_parameters


class TuningErrorBounds(torch.nn.Module):
    """
    Base class for error bounds. This class calculates the real space error and the
    Fourier space error based on the error formula. This class is used in the tuning
    process. It can also be used with the :class:`torchpme.tuning.tuner.TunerBase` to
    build up a custom parameter tuner.

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    """

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


class TunerBase:
    """
    Base class defining the interface for a parameter tuner.

    This class provides a framework for tuning the parameters of a calculator. The class
    itself supports estimating the ``smearing`` from the real space cutoff based on the
    real space error formula. The :func:`TunerBase.tune` defines the interface for a
    sophisticated tuning process, which takes a value of the desired accuracy.

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param cutoff: real space cutoff, serves as a hyperparameter here.
    :param calculator: the calculator to be tuned
    :param exponent: exponent of the potential, only exponent = 1 is supported
    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        cutoff: float,
        calculator: type[Calculator],
        exponent: int = 1,
    ):
        _validate_parameters(charges, cell, positions, exponent)
        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.cutoff = cutoff
        self.calculator = calculator
        self.exponent = exponent
        self._dtype = cell.dtype
        self._device = cell.device

        self._prefac = 2 * float((charges**2).sum()) / math.sqrt(len(positions))

    def tune(self, accuracy: float = 1e-3):
        raise NotImplementedError

    def estimate_smearing(
        self,
        accuracy: float,
    ) -> float:
        """
        Estimate the smearing based on the error formula of the real space. The
        smearing is set as leading to a real space error of ``accuracy/4``.

        :param accuracy: a float, the desired accuracy
        :return: a float, the estimated smearing
        """
        if not isinstance(accuracy, float):
            raise ValueError(f"'{accuracy}' is not a float.")
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
    """
    Tuner using grid search.

    The tuner uses the error formula to estimate the error of a given parameter set.
    If the error is smaller than the accuracy, the timing is measured and returned.
    If the error is larger than the accuracy, the timing is set to infinity and the
    parameter is skipped.

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param cutoff: real space cutoff, serves as a hyperparameter here.
    :param calculator: the calculator to be tuned
    :param params: list of Fourier space parameter sets for which the error is estimated
    :param exponent: exponent of the potential, only exponent = 1 is supported
    :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.Tensor with the pair distances of the neighbors
        for which the potential should be computed in real space.
    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        cutoff: float,
        calculator: type[Calculator],
        error_bounds: type[TuningErrorBounds],
        params: list[dict],
        exponent: int = 1,
        neighbor_indices: Optional[torch.Tensor] = None,
        neighbor_distances: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            charges,
            cell,
            positions,
            cutoff,
            calculator,
            exponent,
        )
        self.error_bounds = error_bounds
        self.params = params
        self.time_func = TuningTimings(
            charges,
            cell,
            positions,
            cutoff,
            neighbor_indices,
            neighbor_distances,
            True,
        )

    def tune(self, accuracy: float = 1e-3) -> tuple[list[float], list[float]]:
        """
        Estimate the error and timing for each parameter set. Only parameters for
        which the error is smaller than the accuracy are timed, the others' timing is
        set to infinity.

        :param accuracy: a float, the desired accuracy
        :return: a list of errors and a list of timings
        """
        if not isinstance(accuracy, float):
            raise ValueError(f"'{accuracy}' is not a float.")
        smearing = self.estimate_smearing(accuracy)
        param_errors = []
        param_timings = []
        for param in self.params:
            error = self.error_bounds(smearing=smearing, cutoff=self.cutoff, **param)  #  type: ignore[call-arg]
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
    """
    Class for timing a calculator.

    The class estimates the average execution time of a given calculater after several
    warmup runs. The class takes the information of the structure that one wants to
    benchmark on, and the configuration of the timing process as inputs.

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param cutoff: real space cutoff, serves as a hyperparameter here.
    :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.Tensor with the pair distances of the neighbors for
        which the potential should be computed in real space.
    :param n_repeat: number of times to repeat to estimate the average timing
    :param n_warmup: number of warmup runs
    :param run_backward: whether to run the backward pass
    """

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

        :param calculator: the calculator to be tuned
        :return: a float, the average execution time
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
