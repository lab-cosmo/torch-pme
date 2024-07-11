"""Test utilities wrap common functions in the metatensor tests"""

from typing import Optional

import pytest
import torch
from vesin import NeighborList


mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")


def add_neighbor_list(
    system, cutoff: Optional[float] = None, full_list: bool = True
) -> None:
    if cutoff is None:
        cell_dimensions = torch.linalg.norm(system.cell, dim=1)
        cutoff_torch = torch.min(cell_dimensions) / 2 - 1e-6
        cutoff = cutoff_torch.item()

    nl = NeighborList(cutoff=cutoff, full_list=full_list)
    i, j, S, D = nl.compute(
        points=system.positions, box=system.cell, periodic=True, quantities="ijSD"
    )

    i = torch.from_numpy(i.astype(int))
    j = torch.from_numpy(j.astype(int))

    neighbor_indices = torch.vstack([i, j])
    neighbor_shifts = torch.from_numpy(S.astype(int))

    sample_values = torch.hstack([neighbor_indices.T, neighbor_shifts])
    samples = mts_torch.Labels(
        names=[
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=sample_values,
    )

    values = torch.from_numpy(D).reshape(-1, 3, 1)
    values = values.type(system.positions.dtype)
    neighbors = mts_torch.TensorBlock(
        values=values,
        samples=samples,
        components=[mts_torch.Labels.range("xyz", 3)],
        properties=mts_torch.Labels.range("distance", 1),
    )

    nl_options = mts_atomistic.NeighborListOptions(cutoff=cutoff, full_list=full_list)
    system.add_neighbor_list(options=nl_options, neighbors=neighbors)
