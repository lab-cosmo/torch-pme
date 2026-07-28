"""
Batched Ewald Computation with Tiling
=====================================

This example demonstrates the *tiled* batched Ewald evaluation, which computes the
potential of many systems of different sizes — and even different periodicities — in a
single :meth:`forward_batched <torchpme.EwaldCalculator.forward_batched>` call. Unlike
the padded approach (see the previous example), atoms are *concatenated* rather than
padded, and the expensive reciprocal-space sum is evaluated in fixed-size tiles over
its block-diagonal system-by-k-vector structure, so no computation is wasted on a
dense ``[n_systems, max_atoms, max_kvectors]`` array.

A single knob, ``num_k`` (the target number of reciprocal-space vectors per system),
balances the real- and reciprocal-space work: it fixes each system's Ewald smearing
and, through it, the real-space cutoff at which its neighbor list must be built. See
:ref:`batched-tiling` for the full design.

The first part of this example uses the pure-torch interface; the second part
evaluates the same systems through the ``metatensor`` interface.
"""

# %%
import time

import torch
import vesin

import torchpme
from torchpme.lib import ewald_params_from_num_k, prepare_tiled_batch

dtype = torch.float64
num_k = 200  # target k-vectors per system: the accuracy/cost knob

# %%
# Build a small heterogeneous batch of 3D-periodic systems with different cells and
# atom counts.

rng = torch.Generator().manual_seed(42)

systems = []
for n_atoms, box in [(2, 3.0), (3, 4.0), (4, 5.0), (5, 6.0), (6, 7.0)]:
    charges = torch.ones(n_atoms, 1, dtype=dtype)
    charges[::2] = -1.0
    charges -= charges.mean()  # neutral cell
    systems.append(
        {
            "positions": torch.rand(n_atoms, 3, generator=rng, dtype=dtype) * box,
            "charges": charges,
            "cell": torch.eye(3, dtype=dtype) * box,
            "periodic": torch.tensor([True, True, True]),
        }
    )

# %%
# The neighbor list of each system is built at the real-space cutoff its cell implies
# for the chosen ``num_k``: bigger cells need a slightly larger cutoff, smaller cells a
# smaller one. :func:`torchpme.lib.ewald_params_from_num_k` derives, per system, the
# reciprocal resolution ``lambda``, the Ewald smearing ``sigma = 2 * lambda``, and the
# neighbor-list cutoff ``r_c = 5 * sigma``. Both factors are adjustable through the
# ``smearing_factor`` and ``cutoff_factor`` arguments, and this can be done once for a
# whole dataset.

for system in systems:
    _, sigma, cutoff = ewald_params_from_num_k(
        system["cell"], system["periodic"], num_k
    )
    neighbor_indices, neighbor_distances = vesin.NeighborList(
        cutoff=float(cutoff), full_list=False
    ).compute(
        points=system["positions"],
        box=system["cell"],
        periodic=True,
        quantities="Pd",
    )
    system["neighbor_indices"] = torch.as_tensor(neighbor_indices, dtype=torch.int64)
    system["neighbor_distances"] = torch.as_tensor(neighbor_distances, dtype=dtype)
    print(
        f"{len(system['charges'])} atoms, box {system['cell'][0, 0]:.0f}: "
        f"sigma = {float(sigma):.3f}, cutoff = {float(cutoff):.3f}"
    )

# %%
# Collate the batch. :func:`torchpme.lib.prepare_tiled_batch` concatenates atoms and
# pairs, derives each system's k-vectors on a common padded grid, and builds the
# static tile-dispatch data. The ``tiling`` dictionary depends only on the cells,
# periodicities, and atom/pair counts — not on the positions — so it is built once per
# batch on the host.

batch, tiling = prepare_tiled_batch(
    positions=[s["positions"] for s in systems],
    charges=[s["charges"] for s in systems],
    cells=[s["cell"] for s in systems],
    periodic=[s["periodic"] for s in systems],
    neighbor_indices=[s["neighbor_indices"] for s in systems],
    neighbor_distances=[s["neighbor_distances"] for s in systems],
    num_k=num_k,
)
print("padded k-vectors per system:", tiling["k_int"].shape[1])
print("per-system smearing:", [f"{s:.3f}" for s in tiling["sigma"].tolist()])

# %%
# Evaluate the whole batch in one call. The calculator's own ``smearing`` and
# ``lr_wavelength`` are overridden by the per-system values carried in the tiling, so
# their construction values do not matter here.

calculator = torchpme.EwaldCalculator(
    torchpme.CoulombPotential(smearing=1.0), lr_wavelength=1.0
)
calculator.to(dtype=dtype)

potentials = calculator.forward_batched(
    batch["charges"],
    batch["cell"],
    batch["positions"],
    batch["neighbor_indices"],
    batch["neighbor_distances"],
    batch["system_index"],
    batch["periodic"],
    tiling,
)
print("concatenated potentials shape:", potentials.shape)

# %%
# The batched result reproduces evaluating each system on its own with the equivalent
# serial parameters (``smearing = sigma_b`` and ``lr_wavelength = sigma_b / 2``).

for index, system in enumerate(systems):
    sigma = float(tiling["sigma"][index])
    serial_calculator = torchpme.EwaldCalculator(
        torchpme.CoulombPotential(smearing=sigma), lr_wavelength=sigma / 2
    )
    serial_calculator.to(dtype=dtype)
    serial = serial_calculator.forward(
        system["charges"],
        system["cell"],
        system["positions"],
        system["neighbor_indices"],
        system["neighbor_distances"],
    )
    mask = batch["system_index"] == index
    torch.testing.assert_close(potentials[mask], serial, rtol=1e-8, atol=1e-10)
print("batched == serial, for every system")

# %%
# Compare the throughput of the batched evaluation against a Python loop over the
# systems.

n_iter = 100

t0 = time.perf_counter()
for _ in range(n_iter):
    calculator.forward_batched(
        batch["charges"],
        batch["cell"],
        batch["positions"],
        batch["neighbor_indices"],
        batch["neighbor_distances"],
        batch["system_index"],
        batch["periodic"],
        tiling,
    )
t_batched = (time.perf_counter() - t0) / n_iter

serial_calculators = []
for index in range(len(systems)):
    sigma = float(tiling["sigma"][index])
    serial_calculator = torchpme.EwaldCalculator(
        torchpme.CoulombPotential(smearing=sigma), lr_wavelength=sigma / 2
    )
    serial_calculator.to(dtype=dtype)
    serial_calculators.append(serial_calculator)

t0 = time.perf_counter()
for _ in range(n_iter):
    for system, serial_calculator in zip(systems, serial_calculators, strict=False):
        serial_calculator.forward(
            system["charges"],
            system["cell"],
            system["positions"],
            system["neighbor_indices"],
            system["neighbor_distances"],
        )
t_loop = (time.perf_counter() - t0) / n_iter

print(f"batched: {t_batched * 1e3:.2f} ms/batch, loop: {t_loop * 1e3:.2f} ms/batch")

# %%
# Evaluation through the metatensor interface
# -------------------------------------------
#
# The same batch can be evaluated with the ``metatensor`` interface. Each structure
# becomes a :class:`metatomic.torch.System` with attached ``"charge"`` data, and each
# neighbor list a :class:`metatensor.torch.TensorBlock` holding the pair *distance
# vectors* (forces flow through them). The static tiling data is built once with
# :func:`torchpme.metatensor.prepare_tiled_batch`.

import metatensor.torch as mts  # noqa: E402
from metatomic.torch import System  # noqa: E402


def to_metatomic_system(system):
    n_atoms = len(system["charges"])
    mta_system = System(
        types=torch.arange(n_atoms, dtype=torch.int32),
        positions=system["positions"],
        cell=system["cell"],
        pbc=system["periodic"],
    )
    charge_block = mts.TensorBlock(
        values=system["charges"],
        samples=mts.Labels.range("atom", n_atoms),
        components=[],
        properties=mts.Labels.range("charge", 1),
    )
    charge_map = mts.TensorMap(
        keys=mts.Labels("_", torch.zeros(1, 1, dtype=torch.int32)),
        blocks=[charge_block],
    )
    mta_system.add_data(name="charge", tensor=charge_map)
    return mta_system


def to_neighbor_block(system):
    # rebuild the list with cell shifts, to get the pair distance *vectors*
    _, _, cutoff = ewald_params_from_num_k(system["cell"], system["periodic"], num_k)
    i, j, shifts = vesin.NeighborList(cutoff=float(cutoff), full_list=False).compute(
        points=system["positions"],
        box=system["cell"],
        periodic=True,
        quantities="ijS",
    )
    indices = torch.stack(
        [torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)],
        dim=1,
    )
    shifts = torch.as_tensor(shifts, dtype=dtype)
    vectors = (
        system["positions"][indices[:, 1]]
        - system["positions"][indices[:, 0]]
        + shifts @ system["cell"]
    )
    samples = torch.zeros(indices.shape[0], 5, dtype=torch.int32)
    samples[:, :2] = indices.to(torch.int32)
    samples[:, 2:] = shifts.to(torch.int32)
    return mts.TensorBlock(
        values=vectors.unsqueeze(-1),
        samples=mts.Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            samples,
        ),
        components=[mts.Labels.range("xyz", 3)],
        properties=mts.Labels.range("distance", 1),
    )


mta_systems = [to_metatomic_system(s) for s in systems]
neighbor_blocks = [to_neighbor_block(s) for s in systems]

mts_tiling = torchpme.metatensor.prepare_tiled_batch(
    mta_systems, neighbor_blocks, num_k=num_k
)

# %%
# The metatensor calculator wraps the same torch calculator; its ``forward_batched``
# takes the systems, the neighbor blocks and the precomputed tiling, and returns a
# :class:`metatensor.torch.TensorMap` whose samples identify each atom by
# ``("system", "atom")``.

mts_calculator = torchpme.metatensor.EwaldCalculator(
    torchpme.CoulombPotential(smearing=1.0), lr_wavelength=1.0
)
mts_calculator.to(dtype=dtype)

result = mts_calculator.forward_batched(mta_systems, neighbor_blocks, mts_tiling)
print(result.block())

# %%
# The metatensor result matches the pure-torch evaluation above.

torch.testing.assert_close(result.block().values, potentials, rtol=1e-12, atol=1e-14)
print("metatensor == pure torch")

# %%
# .. note::
#
#     The batched path also supports 2D-periodic slabs (with the vacuum handled
#     internally through an effective shrunk cell) and non-periodic clusters (summed
#     with the bare, non-range-separated potential) in the same batch — see
#     :ref:`batched-tiling`. On CUDA, the scatter reductions used by the tiled kernel
#     are non-deterministic by default; call
#     ``torch.use_deterministic_algorithms(True)`` for bit-reproducible results.
