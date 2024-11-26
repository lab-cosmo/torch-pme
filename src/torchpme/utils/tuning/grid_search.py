#!/usr/bin/env python3

import time
from typing import Optional

import torch
import vesin.torch
from torchpme import EwaldCalculator, CoulombPotential, PMECalculator, P3MCalculator
from torchpme.utils import tune_ewald, tune_pme, tune_p3m


def grid_search(
    method: str,
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: Optional[float] = None,
    accuracy: float = 1e-3,
    neighbor_indices: Optional[torch.Tensor] = None,
    neighbor_distances: Optional[torch.Tensor] = None,
):
    dtype = charges.dtype
    device = charges.device

    if method == "ewald":
        tune_func = tune_ewald
        CalculatorClass = EwaldCalculator
        all_interpolation_nodes = [1]  # dummy list
    elif method == "pme":
        tune_func = tune_pme
        CalculatorClass = PMECalculator
        all_interpolation_nodes = [3, 4, 5, 6, 7]
    elif method == "p3m":
        tune_func = tune_p3m
        CalculatorClass = P3MCalculator
        all_interpolation_nodes = [2, 3, 4, 5]
    else:
        raise ValueError(f"Invalid method {method}. Choose ewald or pme.")

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    if method == "ewald":
        ns = torch.arange(1, 15, dtype=torch.long, device=device)
        k_space_params = torch.min(cell_dimensions) / ns
    elif method in ["pme", "p3m"]:
        # If you have larger memory, you can try (2, 9)
        ns_actual = torch.exp2(torch.arange(2, 8, dtype=dtype, device=device))
        k_space_params = torch.min(cell_dimensions) / ((ns_actual - 1) / 2)
    else:
        raise ValueError  # Just to make linter happy

    smearing_opt = []
    params_opt = []
    cutoff_opt = []
    time_opt = torch.inf

    for k_space_param in k_space_params:
        for interpolation_nodes in all_interpolation_nodes[::-1]:
            # print(f"Searching for {interpolation_nodes = }, {mesh_spacing = }")
            if method == "ewald":
                smearing, params, cutoff = tune_func(
                    torch.sum(charges**2, dim=0),
                    cell,
                    positions,
                    lr_wavelength=k_space_param,
                    cutoff=cutoff,
                    accuracy=accuracy,
                    learning_rate=0.1,
                    max_steps=10000,
                )
            else:
                smearing, params, cutoff = tune_func(
                    torch.sum(charges**2, dim=0),
                    cell,
                    positions,
                    mesh_spacing=k_space_param,
                    interpolation_nodes=interpolation_nodes,
                    cutoff=cutoff,
                    accuracy=accuracy,
                    learning_rate=0.1,
                    max_steps=10000,
                )
            
            # print(f"{smearing = }, {cutoff = }")

            if cutoff > 10:
                # cutoff too large, no hope to find a solution, even for such a fine
                # mesh, skip
                break

            calculator = CalculatorClass(  # or PMECalculator
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

    return smearing_opt, params_opt, cutoff_opt
