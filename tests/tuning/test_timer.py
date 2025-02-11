import sys
from pathlib import Path

import ase
import torch

from torchpme import (
    CoulombPotential,
    EwaldCalculator,
)
from torchpme.tuning.tuner import TuningTimings

sys.path.append(str(Path(__file__).parents[1]))
from helpers import define_crystal, neighbor_list

DTYPE = torch.float32
DEFAULT_CUTOFF = 4.4
CHARGES_1 = torch.ones((4, 1), dtype=DTYPE)
POSITIONS_1 = 0.3 * torch.arange(12, dtype=DTYPE).reshape((4, 3))
CELL_1 = torch.eye(3, dtype=DTYPE)


def test_timer():
    n_repeat_1 = 10
    n_repeat_2 = 100
    pos, charges, cell, _, _ = define_crystal()

    # use ase to make system bigger
    atoms = ase.Atoms("H" * len(pos), positions=pos.numpy(), cell=cell.numpy())
    atoms.set_initial_charges(charges.numpy().flatten())
    atoms.repeat((4, 4, 4))

    pos = torch.tensor(atoms.positions, dtype=DTYPE)
    charges = torch.tensor(atoms.get_initial_charges(), dtype=DTYPE).reshape(-1, 1)
    cell = torch.tensor(atoms.cell.array, dtype=DTYPE)

    neighbor_indices, neighbor_distances = neighbor_list(
        positions=pos, box=cell, cutoff=DEFAULT_CUTOFF
    )

    calculator = EwaldCalculator(
        potential=CoulombPotential(smearing=1.0),
        lr_wavelength=0.25,
    )

    timing_1 = TuningTimings(
        charges=charges,
        cell=cell,
        positions=pos,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        n_repeat=n_repeat_1,
    )

    timing_2 = TuningTimings(
        charges=charges,
        cell=cell,
        positions=pos,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        n_repeat=n_repeat_2,
    )

    time_1 = timing_1.forward(calculator)
    time_2 = timing_2.forward(calculator)

    assert time_1 > 0
    assert time_1 * n_repeat_1 < time_2 * n_repeat_2
