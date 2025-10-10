import os
import time

import numpy
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from torch.nn.utils.rnn import pad_sequence

from torchpme import CoulombPotential, EwaldCalculator
from torchpme.lib import compute_batched_kvectors

calc = EwaldCalculator(
    potential=CoulombPotential(smearing=1.0),
    full_neighbor_list=True,
    prefactor=1.0,
    lr_wavelength=4.0,
)


xyz_path = os.path.join(os.path.dirname(__file__), "pbc_structures.xyz")
systems = read(xyz_path, index=":")

i_list, j_list, d_list, pos_list, cell_list, charges_list, periodic_list = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)

for atoms in systems:
    i_, j_, d_ = neighbor_list("ijd", atoms, cutoff=5.0)
    i_list.append(torch.tensor(i_, dtype=torch.long))
    j_list.append(torch.tensor(j_, dtype=torch.long))
    d_list.append(torch.tensor(d_, dtype=torch.float32))
    pos_list.append(torch.tensor(atoms.get_positions(), dtype=torch.float32))
    cell_list.append(torch.tensor(numpy.array(atoms.get_cell()), dtype=torch.float32))
    charges_list.append(
        torch.tensor(atoms.get_initial_charges(), dtype=torch.float32) + 1
    )
    periodic_list.append(torch.tensor(atoms.get_pbc(), dtype=torch.bool))

# Pad neighbor indices/distances to the same length for batching
i_batch = pad_sequence(i_list, batch_first=True, padding_value=0)
j_batch = pad_sequence(j_list, batch_first=True, padding_value=0)
d_batch = pad_sequence(d_list, batch_first=True, padding_value=0.0)

# Pair mask: 1 for valid edges, 0 for padded entries
pair_mask = (
    torch.arange(i_batch.shape[1])[None, :]
    < torch.tensor([len(i) for i in i_list])[:, None]
)

# Pad positions and charges to the same number of atoms
max_atoms = max(pos.shape[0] for pos in pos_list)
pos_batch = pad_sequence(pos_list, batch_first=True)
charges_batch = pad_sequence(charges_list, batch_first=True)
cell_batch = torch.stack(cell_list)
periodic_batch = torch.stack(periodic_list)

# Node mask: 1 for valid atoms, 0 for padding
node_mask = (
    torch.arange(max_atoms)[None, :]
    < torch.tensor([p.shape[0] for p in pos_list])[:, None]
)


kvectors = compute_batched_kvectors(lr_wavelength=4.0, cells=cell_batch)


def test_batched_ewald_values():
    values_vmap = torch.vmap(calc.forward)(
        charges_batch.unsqueeze(-1),
        cell_batch,
        pos_batch,
        torch.stack((i_batch, j_batch), dim=-1),
        d_batch,
        periodic_batch,
        node_mask,
        pair_mask,
        kvectors,
    )
    values_loop = []
    for idx in range(len(systems)):
        values_loop.append(
            calc.forward(
                charges_list[idx].unsqueeze(-1),
                cell_list[idx],
                pos_list[idx],
                torch.stack((i_list[idx], j_list[idx]), dim=-1),
                d_list[idx],
                periodic_list[idx],
            )
        )
    values_loop = pad_sequence(values_loop, batch_first=True)
    assert torch.allclose(values_vmap, values_loop, atol=1e-5)


def test_batched_ewald_speed():
    # Time vmap version
    start_batched = time.time()
    _ = torch.vmap(calc.forward)(
        charges_batch.unsqueeze(-1),
        cell_batch,
        pos_batch,
        torch.stack((i_batch, j_batch), dim=-1),
        d_batch,
        periodic_batch,
        node_mask,
        pair_mask,
        kvectors,
    )
    batched_time = time.time() - start_batched

    # Time for-loop version
    start_loop = time.time()
    values_loop = []
    for idx in range(len(systems)):
        values_loop.append(
            calc.forward(
                charges_list[idx].unsqueeze(-1),
                cell_list[idx],
                pos_list[idx],
                torch.stack((i_list[idx], j_list[idx]), dim=-1),
                d_list[idx],
                periodic_list[idx],
            )
        )
    loop_time = time.time() - start_loop
    assert batched_time < loop_time, "Batched version should be faster than loop"
