"""Test utilities wrap common functions in the tests"""

from typing import Optional, Tuple

import torch
from vesin.torch import NeighborList


def neighbor_list_torch(
    positions: torch.tensor,
    periodic: bool = True,
    box: Optional[torch.tensor] = None,
    cutoff: Optional[float] = None,
    full_neighbor_list: bool = False,
) -> Tuple[torch.tensor, torch.tensor]:

    if box is None:
        box = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)

    if cutoff is None:
        cell_dimensions = torch.linalg.norm(box, dim=1)
        cutoff_torch = torch.min(cell_dimensions) / 2 - 1e-6
        cutoff = cutoff_torch.item()

    nl = NeighborList(cutoff=cutoff, full_list=full_neighbor_list)
    i, j, d = nl.compute(points=positions, box=box, periodic=periodic, quantities="ijd")

    neighbor_indices = torch.stack([i, j], dim=1)

    return neighbor_indices, d
