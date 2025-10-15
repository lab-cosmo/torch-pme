"""
Batched Ewald Computation with Padding
====================================

This example demonstrates how to compute Ewald potentials for a batch of systems
with different numbers of atoms using padding. The idea is to pad atomic positions,
charges, and neighbor lists to the same length and use masks to ignore padded entries
during computation.
"""

# %%
import torch
import vesin
from torch.nn.utils.rnn import pad_sequence

import torchpme

dtype = torch.float64
cutoff = 4.4

# %%
# Example: two systems with different numbers of atoms
systems = [
    {
        "symbols": ("Cs", "Cl"),
        "positions": torch.tensor([(0, 0, 0), (0.5, 0.5, 0.5)], dtype=dtype),
        "charges": torch.tensor([[1.0], [-1.0]], dtype=dtype),
        "cell": torch.eye(3, dtype=dtype) * 3.0,
        "pbc": torch.tensor([True, True, True]),
    },
    {
        "symbols": ("Na", "Cl", "Cl"),
        "positions": torch.tensor(
            [(0, 0, 0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)], dtype=dtype
        ),
        "charges": torch.tensor([[1.0], [-1.0], [-1.0]], dtype=dtype),
        "cell": torch.eye(3, dtype=dtype) * 4.0,
        "pbc": torch.tensor([True, True, True]),
    },
]

# %%
# Compute neighbor lists for each system
i_list, j_list, d_list, pos_list, charges_list, cell_list, periodic_list = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)

nl = vesin.NeighborList(cutoff=cutoff, full_list=False)

for sys in systems:
    neighbor_indices, neighbor_distances = nl.compute(
        points=sys["positions"],
        box=sys["cell"],
        periodic=sys["pbc"][0],
        quantities="Pd",
    )
    i_list.append(torch.tensor(neighbor_indices[:, 0], dtype=torch.int64))
    j_list.append(torch.tensor(neighbor_indices[:, 1], dtype=torch.int64))
    d_list.append(torch.tensor(neighbor_distances, dtype=dtype))
    pos_list.append(sys["positions"])
    charges_list.append(sys["charges"])
    cell_list.append(sys["cell"])
    periodic_list.append(sys["pbc"])

# %%
# Pad positions, charges, and neighbor lists
max_atoms = max(pos.shape[0] for pos in pos_list)
pos_batch = pad_sequence(pos_list, batch_first=True)
charges_batch = pad_sequence(charges_list, batch_first=True)
cell_batch = torch.stack(cell_list)
periodic_batch = torch.stack(periodic_list)
i_batch = pad_sequence(i_list, batch_first=True, padding_value=0)
j_batch = pad_sequence(j_list, batch_first=True, padding_value=0)
d_batch = pad_sequence(d_list, batch_first=True, padding_value=0.0)

# Masks for ignoring padded atoms and neighbor entries
node_mask = (
    torch.arange(max_atoms)[None, :]
    < torch.tensor([p.shape[0] for p in pos_list])[:, None]
)
pair_mask = (
    torch.arange(i_batch.shape[1])[None, :]
    < torch.tensor([len(i) for i in i_list])[:, None]
)
# %%
# Initialize Ewald calculator
calculator = torchpme.EwaldCalculator(
    torchpme.CoulombPotential(smearing=0.5),
    lr_wavelength=4.0,
)
calculator.to(dtype=dtype)

# %%
# Compute potentials in a batched manner using vmap
kvectors = torchpme.lib.compute_batched_kvectors(
    lr_wavelength=calculator.lr_wavelength, cells=cell_batch
)

potentials_batch = torch.vmap(calculator.forward)(
    charges_batch,
    cell_batch,
    pos_batch,
    torch.stack((i_batch, j_batch), dim=-1),
    d_batch,
    periodic_batch,
    node_mask,
    pair_mask,
    kvectors,
)

# %%
print("Batched potentials shape:", potentials_batch.shape)
print(potentials_batch)
# %%
