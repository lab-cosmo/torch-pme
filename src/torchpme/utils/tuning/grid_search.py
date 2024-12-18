#!/usr/bin/env python3

import math
import time
from typing import Optional
from warnings import warn

import torch
import vesin.torch

from torchpme import (
    CoulombPotential,
    EwaldCalculator,
    P3MCalculator,
    PMECalculator,
)
from torchpme.utils import EwaldErrorBounds, P3MErrorBounds, PMEErrorBounds

from . import _estimate_smearing_cutoff, _validate_parameters


def grid_search(
    method: str,
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    exponent: int = 1,
    accuracy: float = 1e-3,
    neighbor_indices: Optional[torch.Tensor] = None,
    neighbor_distances: Optional[torch.Tensor] = None,
):
    r"""
    Find the optimal parameters for calculators.

    The steps are:
    1. Find the ``smearing`` parameter for the :py:class:`CoulombPotential` that leads
    to a real space error of half the desired accuracy.
    2. Grid search for the kspace parameters, i.e. the ``lr_wavelength`` for Ewald and
    the ``mesh_spacing`` and ``interpolation_nodes`` for PME and P3M.
    For each combination of parameters, calculate the error. If the error is smaller
    than the desired accuracy, use this combination for test runs to get the calculation
    time needed. Return the combination that leads to the shortest calculation time. If
    the desired accuracy is never reached, return the combination that leads to the
    smallest error and throw a warning.

    :param charges: torch.Tensor, atomic (pseudo-)charges
    :param cell: torch.Tensor, periodic supercell for the system
    :param positions: torch.Tensor, Cartesian coordinates of the particles within
        the supercell.
    :param cutoff: float, cutoff distance for the neighborlist
    :param exponent :math:`p` in :math:`1/r^p` potentials
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.
    :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.Tensor with the pair distances of the neighbors
        for which the potential should be computed in real space.

    :return: Tuple containing a float of the optimal smearing for the :py:class:
    `CoulombPotential`, a dictionary with the parameters for the calculator of the
    chosen method and a float of the optimal cutoff value for the neighborlist
    computation.
    """
    dtype = charges.dtype
    device = charges.device
    _validate_parameters(charges, cell, positions, exponent, accuracy)

    if method == "ewald":
        err_func = EwaldErrorBounds(
            sum_squared_charges=torch.sum(charges**2, dim=0),
            cell=cell,
            positions=positions,
        )
        CalculatorClass = EwaldCalculator
        all_interpolation_nodes = [1]  # dummy list
    elif method == "pme":
        err_func = PMEErrorBounds(
            sum_squared_charges=torch.sum(charges**2, dim=0),
            cell=cell,
            positions=positions,
        )
        CalculatorClass = PMECalculator
        all_interpolation_nodes = [3, 4, 5, 6, 7]
    elif method == "p3m":
        err_func = P3MErrorBounds(
            sum_squared_charges=torch.sum(charges**2, dim=0),
            cell=cell,
            positions=positions,
        )
        CalculatorClass = P3MCalculator
        all_interpolation_nodes = [2, 3, 4, 5]
    else:
        raise ValueError(f"Invalid method {method}. Choose ewald or pme.")

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    if method == "ewald":
        ns = torch.arange(1, 15, dtype=torch.long, device=device)
        k_space_params = torch.min(cell_dimensions) / ns
    elif method in ["pme", "p3m"]:
        ns_actual = torch.exp2(torch.arange(2, 8, dtype=dtype, device=device))
        k_space_params = torch.min(cell_dimensions) / ((ns_actual - 1) / 2)
    else:
        raise ValueError  # Just to make linter happy

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
        cell,
        smearing=None,
        cutoff=cutoff,
        accuracy=accuracy,
        prefac=2 * (charges**2).sum() / math.sqrt(len(positions)),
    )
    for k_space_param in k_space_params:
        for interpolation_nodes in all_interpolation_nodes[::-1]:
            if method == "ewald":
                params = {
                    "lr_wavelength": float(k_space_param),
                }
            else:
                params = {
                    "mesh_spacing": float(k_space_param),
                    "interpolation_nodes": int(interpolation_nodes),
                }

            err = err_func(
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

            calculator = CalculatorClass(
                potential=CoulombPotential(smearing=smearing),
                **params,
            )
            if neighbor_indices is None and neighbor_distances is None:
                nl = vesin.torch.NeighborList(cutoff=cutoff, full_list=False)
                i, j, neighbor_distances = nl.compute(
                    points=positions.to(dtype=torch.float64, device="cpu"),
                    box=cell.to(dtype=torch.float64, device="cpu"),
                    periodic=True,
                    quantities="ijd",
                )
                neighbor_indices = torch.stack([i, j], dim=1)
            elif neighbor_indices is None or neighbor_distances is None:
                raise ValueError(
                    "If neighbor_indices or neighbor_distances are None, "
                    "both must be None."
                )
            neighbor_indices = neighbor_indices.to(device=device)
            neighbor_distances = neighbor_distances.to(dtype=dtype, device=device)

            # warm-up
            for _ in range(5):
                calculator.forward(
                    positions=positions,
                    charges=charges,
                    cell=cell,
                    neighbor_indices=neighbor_indices,
                    neighbor_distances=neighbor_distances,
                )

            # measure time
            t0 = time.time()
            calculator.forward(
                positions=positions,
                charges=charges,
                cell=cell,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
            )
            if device is torch.device("cuda"):
                torch.cuda.synchronize()
            execution_time = time.time() - t0

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
