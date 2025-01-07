#!/usr/bin/env python3

import math
from copy import copy
from itertools import product
from typing import Optional
from warnings import warn

import torch

from ...calculators import (
    Calculator,
)
from ...potentials import InversePowerLawPotential
from . import (
    TuningErrorBounds,
    TuningTimings,
)


class GridSearchBase:
    r"""
    Base class for finding the optimal parameters for calculators using a grid search.

    :param charges: torch.Tensor, atomic (pseudo-)charges
    :param cell: torch.Tensor, periodic supercell for the system
    :param positions: torch.Tensor, Cartesian coordinates of the particles within
        the supercell.
    :param cutoff: float, cutoff distance for the neighborlist
    :param exponent :math:`p` in :math:`1/r^p` potentials, currently only :math:`p=1` is
        supported
    :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.Tensor with the pair distances of the neighbors
        for which the potential should be computed in real space.
    """

    ErrorBounds: type[TuningErrorBounds]
    Timings: type[TuningTimings] = TuningTimings
    CalculatorClass: type[Calculator]
    TemplateGridSearchParams: dict[
        str, torch.Tensor
    ]  # {"interpolation_nodes": ..., ...}

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        cutoff: float,
        exponent: int = 1,
        neighbor_indices: Optional[torch.Tensor] = None,
        neighbor_distances: Optional[torch.Tensor] = None,
    ):
        self._validate_parameters(charges, cell, positions, exponent)
        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.cutoff = cutoff
        self.exponent = exponent
        self.dtype = charges.dtype
        self.device = charges.device
        self.err_func = self.ErrorBounds(charges, cell, positions)
        self._cell_dimensions = torch.linalg.norm(cell, dim=1)
        self.time_func = self.Timings(
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

        self._prefac = 2 * (charges**2).sum() / math.sqrt(len(positions))
        self.GridSearchParams = copy(self.TemplateGridSearchParams)

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

    def _estimate_smearing(
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
        smearing_init = self.cutoff / ratio

        return float(smearing_init)

    def tune(
        self,
        accuracy: float = 1e-3,
    ):
        r"""
        The steps are: 1. Find the ``smearing`` parameter for the
        :py:class:`CoulombPotential` that leads to a real space error of half the
        desired accuracy. 2. Grid search for the kspace parameters, i.e. the
        ``lr_wavelength`` for Ewald and the ``mesh_spacing`` and ``interpolation_nodes``
        for PME and P3M. For each combination of parameters, calculate the error. If the
        error is smaller than the desired accuracy, use this combination for test runs
        to get the calculation time needed. Return the combination that leads to the
        shortest calculation time. If the desired accuracy is never reached, return the
        combination that leads to the smallest error and throw a warning.

        :param accuracy: Recomended values for a balance between the accuracy and speed
            is :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.

        :return: Tuple containing a float of the optimal smearing for the :py:class:
        `CoulombPotential`, a dictionary with the parameters for the calculator of the
        chosen method and a float of the optimal cutoff value for the neighborlist
        computation.
        """
        if not isinstance(accuracy, float):
            raise ValueError(f"'{accuracy}' is not a float.")

        smearing_opt = None
        params_opt = None
        cutoff_opt = None
        time_opt = torch.inf

        # In case there is no parameter reaching the accuracy, return
        # the best found so far
        smearing_err_opt = None
        params_err_opt = None
        cutoff_err_opt = None
        err_opt = torch.inf

        smearing = self._estimate_smearing(
            accuracy=accuracy,
        )
        cutoff = self.cutoff
        for param_values in product(*self.GridSearchParams.values()):
            params = dict(zip(self.GridSearchParams.keys(), param_values))
            err = self.err_func(
                smearing=smearing,
                cutoff=cutoff,
                **params,
            )

            if err > accuracy:
                # Not going to test the time, record the parameters if the error is
                # better.
                if err < err_opt:
                    smearing_err_opt = smearing
                    params_err_opt = params
                    cutoff_err_opt = cutoff
                    err_opt = err
                continue

            execution_time = self._timing(smearing, params)
            if execution_time < time_opt:
                smearing_opt = smearing
                params_opt = params
                cutoff_opt = cutoff
                time_opt = execution_time

        if time_opt == torch.inf:
            # Never found a parameter that reached the accuracy
            warn(
                f"No parameters found within the desired accuracy of {accuracy}."
                f"Returning the best found. Accuracy: {str(err_opt)}",
                stacklevel=1,
            )
            return smearing_err_opt, params_err_opt, cutoff_err_opt

        return smearing_opt, params_opt, cutoff_opt

    def _timing(self, smearing: float, params: dict):
        calculator = self.CalculatorClass(
            potential=InversePowerLawPotential(
                exponent=self.exponent,  # but only exponent = 1 is supported
                smearing=smearing,
            ),
            **params,
        )

        return self.time_func(calculator)
