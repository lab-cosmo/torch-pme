#!/usr/bin/env python3

import math
import time
from itertools import product
from typing import Optional
from warnings import warn

import torch
import vesin.torch

from ...calculators import (
    Calculator,
)
from ...potentials import CoulombPotential
from . import (
    TuningErrorBounds,
    _estimate_smearing_cutoff,
    _validate_parameters,
)


class GridSearchBase:
    r"""
    Base class for finding the optimal parameters for calculators using a grid search.

    :param charges: torch.Tensor, atomic (pseudo-)charges
    :param cell: torch.Tensor, periodic supercell for the system
    :param positions: torch.Tensor, Cartesian coordinates of the particles within
        the supercell.
    :param cutoff: float, cutoff distance for the neighborlist
    :param exponent :math:`p` in :math:`1/r^p` potentials
    :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.Tensor with the pair distances of the neighbors
        for which the potential should be computed in real space.
    """

    ErrorBounds: type[TuningErrorBounds]
    CalculatorClass: type[Calculator]
    GridSearchParams: dict[str, torch.Tensor]  # {"interpolation_nodes": ..., ...}

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
        _validate_parameters(charges, cell, positions, exponent, 1e-1)
        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.cutoff = cutoff
        self.dtype = charges.dtype
        self.device = charges.device
        self.err_func = self.ErrorBounds(charges, cell, positions)
        self._cell_dimensions = torch.linalg.norm(cell, dim=1)

        self._prefac = 2 * (charges**2).sum() / math.sqrt(len(positions))

        if neighbor_indices is None and neighbor_distances is None:
            nl = vesin.torch.NeighborList(cutoff=cutoff, full_list=False)
            i, j, neighbor_distances = nl.compute(
                points=self.positions.to(dtype=torch.float64, device="cpu"),
                box=self.cell.to(dtype=torch.float64, device="cpu"),
                periodic=True,
                quantities="ijd",
            )
            neighbor_indices = torch.stack([i, j], dim=1)
        elif neighbor_indices is None or neighbor_distances is None:
            raise ValueError(
                "If neighbor_indices or neighbor_distances are None, "
                "both must be None."
            )
        self.neighbor_indices = neighbor_indices.to(device=self.device)
        self.neighbor_distances = neighbor_distances.to(
            dtype=self.dtype, device=self.device
        )

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

        smearing, cutoff = _estimate_smearing_cutoff(
            self.cell,
            smearing=None,
            cutoff=self.cutoff,
            accuracy=accuracy,
            prefac=self._prefac,
        )
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
            potential=CoulombPotential(smearing=smearing),
            **params,
        )
        # warm-up
        for _ in range(5):
            calculator.forward(
                positions=self.positions,
                charges=self.charges,
                cell=self.cell,
                neighbor_indices=self.neighbor_indices,
                neighbor_distances=self.neighbor_distances,
            )

        # measure time
        t0 = time.time()
        calculator.forward(
            positions=self.positions,
            charges=self.charges,
            cell=self.cell,
            neighbor_indices=self.neighbor_indices,
            neighbor_distances=self.neighbor_distances,
        )
        if self.device is torch.device("cuda"):
            torch.cuda.synchronize()

        return time.time() - t0
