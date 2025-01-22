import math
import time
from typing import Optional

import torch

from ..calculators import Calculator
from ..potentials import InversePowerLawPotential
from ._utils import _validate_parameters


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

    def error(self, *args, **kwargs):
        raise NotImplementedError


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

    Example
    -------
    >>> import torch
    >>> import torchpme
    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)
    >>> tuner = TunerBase(charges, cell, positions, 4.4, torchpme.EwaldCalculator)
    >>> smearing = tuner.estimate_smearing(1e-3)
    >>> print(smearing)
    1.1069526756106463

    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        cutoff: float,
        calculator: type[Calculator],
        exponent: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        _validate_parameters(charges, cell, positions, exponent)
        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.cutoff = cutoff
        self.calculator = calculator
        self.exponent = exponent
        self.device = "cpu" if device is None else device
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

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

    The tuner uses the error formula to estimate the error of a given parameter set. If
    the error is smaller than the accuracy, the timing is measured and returned. If the
    error is larger than the accuracy, the timing is set to infinity and the parameter
    is skipped.

    .. note::

        The cutoff is treated as a hyperparameter here. In case one wants to tune the
        cutoff, one could instantiate the tuner with different cutoff values and
        manually pick the best from the tuning results.

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param cutoff: real space cutoff, serves as a hyperparameter here.
    :param calculator: the calculator to be tuned
    :param error_bounds: error bounds for the calculator
    :param params: list of Fourier space parameter sets for which the error is estimated
    :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.Tensor with the pair distances of the neighbors for
        which the potential should be computed in real space.
    :param exponent: exponent of the potential, only exponent = 1 is supported
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
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
        exponent: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            charges=charges,
            cell=cell,
            positions=positions,
            cutoff=cutoff,
            calculator=calculator,
            exponent=exponent,
            dtype=dtype,
            device=device,
        )
        self.error_bounds = error_bounds
        self.params = params
        self.time_func = TuningTimings(
            charges,
            cell,
            positions,
            neighbor_indices,
            neighbor_distances,
            True,
            dtype=dtype,
            device=device,
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
            error = self.error_bounds(smearing=smearing, cutoff=self.cutoff, **param)  # type: ignore[call-arg]
            param_errors.append(float(error))
            # only computes timings for parameters that meet the accuracy requirements
            param_timings.append(
                self._timing(smearing, param) if error <= accuracy else float("inf")
            )

        return param_errors, param_timings

    def _timing(self, smearing: float, k_space_params: dict):
        calculator = self.calculator(
            potential=InversePowerLawPotential(
                exponent=self.exponent,  # but only exponent = 1 is supported
                smearing=smearing,
                device=self.device,
                dtype=self.dtype,
            ),
            device=self.device,
            dtype=self.dtype,
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
    :param n_warmup: number of warmup runs, recommended to be at least 4
    :param run_backward: whether to run the backward pass
    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
        n_repeat: int = 4,
        n_warmup: int = 4,
        run_backward: Optional[bool] = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.dtype = dtype
        self.device = device
        self.n_repeat = n_repeat
        self.n_warmup = n_warmup
        self.run_backward = run_backward
        self.neighbor_indices = neighbor_indices.to(device=self.device)
        self.neighbor_distances = neighbor_distances.to(
            dtype=self.dtype, device=self.device
        )

    def forward(self, calculator: torch.nn.Module):
        """
        Estimate the execution time of a given calculator for the structure
        to be used as benchmark.

        :param calculator: the calculator to be tuned
        :return: a float, the average execution time
        """
        # measure time
        execution_time = 0.0

        for _ in range(self.n_repeat + self.n_warmup):
            if _ == self.n_warmup:
                execution_time = 0.0
            positions = self.positions.clone()
            cell = self.cell.clone()
            charges = self.charges.clone()
            # nb - this won't compute gradiens involving the distances
            if self.run_backward:
                positions.requires_grad_(True)
                cell.requires_grad_(True)
                charges.requires_grad_(True)
            execution_time -= time.monotonic()
            result = calculator.forward(
                positions=positions,
                charges=charges,
                cell=cell,
                neighbor_indices=self.neighbor_indices,
                neighbor_distances=self.neighbor_distances,
            )
            value = result.sum()
            if self.run_backward:
                value.backward(retain_graph=True)

            if self.device is torch.device("cuda"):
                torch.cuda.synchronize()
            execution_time += time.monotonic()

        return execution_time / self.n_repeat
