import math
import time
from typing import Optional

import torch

from .._utils import _validate_parameters
from ..calculators import Calculator
from ..potentials import InversePowerLawPotential


class TuningErrorBounds(torch.nn.Module):
    """
    Base class for error bounds. This class calculates the real space error and the
    Fourier space error based on the error formula. This class is used in the tuning
    process. It can also be used with the :class:`torchpme.tuning.tuner.TunerBase` to
    build up a custom parameter tuner.

    :param charges: torch.tensor of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
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

    :param charges: torch.tensor of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
    :param cutoff: real space cutoff, serves as a hyperparameter here.
    :param calculator: the calculator to be tuned
    :param exponent: exponent of the potential, only exponent = 1 is supported
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.

    Example
    -------
    >>> import torch
    >>> import torchpme
    >>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]])
    >>> charges = torch.tensor([[1.0], [-1.0]])
    >>> cell = torch.eye(3)
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
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
    ):
        if exponent != 1:
            raise NotImplementedError(
                f"Only exponent = 1 is supported but got {exponent}."
            )

        _validate_parameters(
            charges=charges,
            cell=cell,
            positions=positions,
            neighbor_indices=torch.tensor([[0, 1]], device=positions.device),
            neighbor_distances=torch.tensor(
                [1.0], device=positions.device, dtype=positions.dtype
            ),
            smearing=1.0,  # dummy value because; always have range-seperated potentials
        )
        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.cutoff = cutoff
        self.calculator = calculator
        self.exponent = exponent
        self.full_neighbor_list = full_neighbor_list
        self.prefactor = prefactor

        self._smearing_esti_prefac = (
            2 * float((charges**2).sum()) / math.sqrt(len(positions))
        )

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
                / self._smearing_esti_prefac
                * math.sqrt(self.cutoff * float(torch.abs(self.cell.det())))
            )
        )
        smearing = self.cutoff / ratio

        return float(smearing)

    @staticmethod
    def filter_neighbors(
        cutoff: float, neighbor_indices: torch.Tensor, neighbor_distances: torch.Tensor
    ):
        """
        Filter neighbor indices and distances based on a user given cutoff. This allows
        users pre-computing the neighbor list with a larger cutoff and then filtering
        the neighbors based on a smaller cutoff, leading to a faster tuning on the
        cutoff.

        :param cutoff: real space cutoff
        :param neighbor_indices: torch.tensor with the ``i,j`` indices of neighbors for
            which the potential should be computed in real space.
        :param neighbor_distances: torch.tensor with the pair distances of the neighbors
            for which the potential should be computed in real space.
        """
        filter_idx = torch.where(neighbor_distances < cutoff)
        return neighbor_indices[filter_idx], neighbor_distances[filter_idx]


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

    :param charges: torch.tensor of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
    :param cutoff: real space cutoff, serves as a hyperparameter here.
    :param calculator: the calculator to be tuned
    :param error_bounds: error bounds for the calculator
    :param params: list of Fourier space parameter sets for which the error is estimated
    :param neighbor_indices: torch.tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.tensor with the pair distances of the neighbors for
        which the potential should be computed in real space.
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
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
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
        exponent: int = 1,
    ):
        super().__init__(
            charges=charges,
            cell=cell,
            positions=positions,
            cutoff=cutoff,
            calculator=calculator,
            exponent=exponent,
            full_neighbor_list=full_neighbor_list,
            prefactor=prefactor,
        )
        self.error_bounds = error_bounds
        self.params = params
        neighbor_indices, neighbor_distances = self.filter_neighbors(
            cutoff, neighbor_indices, neighbor_distances
        )
        self.time_func = TuningTimings(
            charges,
            cell,
            positions,
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
            ),
            full_neighbor_list=self.full_neighbor_list,
            prefactor=self.prefactor,
            **k_space_params,
        )
        calculator.to(device=self.positions.device, dtype=self.positions.dtype)
        return self.time_func(calculator)


class TuningTimings(torch.nn.Module):
    """
    Class for timing a calculator.

    The class estimates the average execution time of a given calculater after several
    warmup runs. The class takes the information of the structure that one wants to
    benchmark on, and the configuration of the timing process as inputs.

    :param charges: torch.tensor of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
    :param cutoff: real space cutoff, serves as a hyperparameter here.
    :param neighbor_indices: torch.tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.tensor with the pair distances of the neighbors for
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
    ):
        super().__init__()

        _validate_parameters(
            charges=charges,
            cell=cell,
            positions=positions,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            smearing=1.0,  # dummy value because; always have range-seperated potentials
        )

        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.n_repeat = n_repeat
        self.n_warmup = n_warmup
        self.run_backward = run_backward
        self.neighbor_indices = neighbor_indices
        self.neighbor_distances = neighbor_distances

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

            execution_time += time.monotonic()

        return execution_time / self.n_repeat
