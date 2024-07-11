"""Test utilities wrap common functions in the tests"""

from typing import Optional, Tuple

import torch
from vesin import NeighborList


def neighbor_list_torch(
    positions: torch.tensor, cell: torch.tensor, cutoff: Optional[float] = None
) -> Tuple[torch.tensor, torch.tensor]:

    if cutoff is None:
        cell_dimensions = torch.linalg.norm(cell, dim=1)
        cutoff_torch = torch.min(cell_dimensions) / 2 - 1e-6
        cutoff = cutoff_torch.item()

    nl = NeighborList(cutoff=cutoff, full_list=True)
    i, j, S = nl.compute(points=positions, box=cell, periodic=True, quantities="ijS")

    i = torch.from_numpy(i.astype(int))
    j = torch.from_numpy(j.astype(int))

    neighbor_indices = torch.vstack([i, j])
    neighbor_shifts = torch.from_numpy(S.astype(int))

    return neighbor_indices, neighbor_shifts
