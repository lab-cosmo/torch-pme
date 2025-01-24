import sys
from pathlib import Path

import torch

from torchpme import (
    CoulombPotential,
    EwaldCalculator,
)
from torchpme.tuning.tuner import TuningTimings

sys.path.append(str(Path(__file__).parents[1]))
from helpers import compute_distances, define_crystal, neighbor_list

DTYPE = torch.float32
DEFAULT_CUTOFF = 4.4
CHARGES_1 = torch.ones((4, 1), dtype=DTYPE)
POSITIONS_1 = 0.3 * torch.arange(12, dtype=DTYPE).reshape((4, 3))
CELL_1 = torch.eye(3, dtype=DTYPE)


def _nl_calculation(pos, cell):
    neighbor_indices, neighbor_shifts = neighbor_list(
        positions=pos,
        periodic=True,
        box=cell,
        cutoff=DEFAULT_CUTOFF,
        neighbor_shifts=True,
    )

    neighbor_distances = compute_distances(
        positions=pos,
        neighbor_indices=neighbor_indices,
        cell=cell,
        neighbor_shifts=neighbor_shifts,
    )

    return neighbor_indices, neighbor_distances


def test_timer():
    n_repeat_1 = 8
    n_repeat_2 = 16
    pos, charges, cell, madelung_ref, num_units = define_crystal()
    neighbor_indices, neighbor_distances = _nl_calculation(pos, cell)

    calculator = EwaldCalculator(
        potential=CoulombPotential(smearing=1.0),
        lr_wavelength=0.25,
        dtype=DTYPE,
    )

    timing_1 = TuningTimings(
        charges=charges,
        cell=cell,
        positions=pos,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        dtype=DTYPE,
        n_repeat=n_repeat_1,
    )

    timing_2 = TuningTimings(
        charges=charges,
        cell=cell,
        positions=pos,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        dtype=DTYPE,
        n_repeat=n_repeat_2,
    )

    time_1 = timing_1.forward(calculator)
    time_2 = timing_2.forward(calculator)

    assert time_1 > 0
    assert time_1 * n_repeat_1 < time_2 * n_repeat_2
