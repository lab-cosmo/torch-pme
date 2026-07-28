"""
Tests for the host-side collation helpers in :mod:`torchpme.lib.batching`.

These build the inputs of :meth:`EwaldCalculator.forward_batched` and are therefore
checked here directly — per-system Ewald parameters, the 2D cell shrink, the integer
k-grids and the structural invariants of the tiling — rather than only through the
values a batched evaluation produces (see ``tests/calculators/test_batched_ewald.py``
for those).
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from torchpme.lib import ewald_params_from_num_k, prepare_tiled_batch, shrink_2d_cell
from torchpme.lib.batching import _integer_kgrid

sys.path.append(str(Path(__file__).parents[1]))
from helpers import DEVICES

DTYPE = torch.float64
NUM_K = 200


def cluster(n_atoms=5, seed=3, dtype=DTYPE, device="cpu"):
    """A non-periodic system with an all-pairs (half) neighbor list."""
    generator = torch.Generator().manual_seed(seed)
    positions = torch.rand((n_atoms, 3), generator=generator, dtype=dtype) * 3.0
    charges = torch.ones(n_atoms, 1, dtype=dtype)
    charges[::2] = -1.0
    indices = torch.triu_indices(n_atoms, n_atoms, offset=1).T
    distances = (positions[indices[:, 0]] - positions[indices[:, 1]]).norm(dim=-1)
    return {
        "positions": positions.to(device),
        "charges": charges.to(device),
        "cell": torch.zeros(3, 3, dtype=dtype, device=device),
        "periodic": torch.tensor([False, False, False], device=device),
        "neighbor_indices": indices.to(device),
        "neighbor_distances": distances.to(device),
    }


def crystal(box=4.0, n_atoms=4, seed=7, periodic=(True, True, True), dtype=DTYPE):
    """A periodic system with a (physically incomplete) placeholder pair list."""
    sample = cluster(n_atoms=n_atoms, seed=seed, dtype=dtype)
    sample["cell"] = box * torch.eye(3, dtype=dtype)
    sample["periodic"] = torch.tensor(periodic)
    return sample


def slab(vacuum=20.0, dtype=DTYPE):
    sample = crystal(periodic=(True, True, False), dtype=dtype)
    sample["cell"] = torch.diag(torch.tensor([4.0, 4.0, vacuum], dtype=dtype))
    return sample


def collate(samples, **kwargs):
    kwargs.setdefault("k_pad_fraction", 1.0)
    return prepare_tiled_batch(
        positions=[s["positions"] for s in samples],
        charges=[s["charges"] for s in samples],
        cells=[s["cell"] for s in samples],
        periodic=[s["periodic"] for s in samples],
        neighbor_indices=[s["neighbor_indices"] for s in samples],
        neighbor_distances=[s["neighbor_distances"] for s in samples],
        num_k=kwargs.pop("num_k", NUM_K),
        **kwargs,
    )


# --- shrink_2d_cell -----------------------------------------------------------------


def test_shrink_leaves_3d_and_nonperiodic_cells_alone():
    cell = 4.0 * torch.eye(3, dtype=DTYPE)
    positions = torch.zeros(2, 3, dtype=DTYPE)
    for periodic in ([True, True, True], [False, False, False]):
        shrunk = shrink_2d_cell(cell, torch.tensor(periodic), positions)
        assert shrunk is cell


def test_shrink_leaves_short_vacuum_alone():
    # thickness 1.5 + 1.5 * 4.0 = 7.5 is the smallest accepted height
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=DTYPE)
    periodic = torch.tensor([True, True, False])
    cell = torch.diag(torch.tensor([4.0, 4.0, 7.0], dtype=DTYPE))
    assert shrink_2d_cell(cell, periodic, positions) is cell


def test_shrink_replaces_large_vacuum():
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=DTYPE)
    periodic = torch.tensor([True, True, False])
    cell = torch.diag(torch.tensor([4.0, 4.0, 100.0], dtype=DTYPE))
    shrunk = shrink_2d_cell(cell, periodic, positions)
    assert_close(shrunk[:2], cell[:2])
    assert_close(shrunk[2], torch.tensor([0.0, 0.0, 7.5], dtype=DTYPE))


def test_shrink_synthesizes_zero_vacuum_row():
    # the metatomic convention for non-periodic directions is a zero cell vector
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=DTYPE)
    periodic = torch.tensor([True, True, False])
    cell = torch.diag(torch.tensor([4.0, 4.0, 0.0], dtype=DTYPE))
    shrunk = shrink_2d_cell(cell, periodic, positions)
    assert_close(shrunk[:2], cell[:2])
    assert_close(shrunk[2], torch.tensor([0.0, 0.0, 7.5], dtype=DTYPE))
    # the synthesized axis is normal to the periodic plane
    assert_close(shrunk[2] @ shrunk[0], torch.tensor(0.0, dtype=DTYPE))
    assert_close(shrunk[2] @ shrunk[1], torch.tensor(0.0, dtype=DTYPE))


def test_shrink_keeps_the_direction_of_a_tilted_vacuum_row():
    # slab cells in the wild are not always orthogonal, e.g. the MAD subsets
    cell = torch.tensor(
        [[11.0, 2.1, 0.0], [0.0, 9.9, 0.0], [-0.4, 2.3, 100.0]], dtype=DTYPE
    )
    periodic = torch.tensor([True, True, False])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=DTYPE)
    shrunk = shrink_2d_cell(cell, periodic, positions)

    direction = cell[2] / torch.linalg.norm(cell[2])
    assert_close(shrunk[2] / torch.linalg.norm(shrunk[2]), direction)
    thickness = float((positions @ direction).max() - (positions @ direction).min())
    l_max = float(torch.linalg.norm(cell[:2], dim=-1).max())
    assert_close(
        torch.linalg.norm(shrunk[2]),
        torch.tensor(thickness + 1.5 * l_max, dtype=DTYPE),
    )


# --- ewald_params_from_num_k --------------------------------------------------------


def test_params_single_matches_batched():
    cells = torch.stack(
        [4.0 * torch.eye(3, dtype=DTYPE), 6.0 * torch.eye(3, dtype=DTYPE)]
    )
    periodic = torch.tensor([[True, True, True], [True, True, True]])
    batched = ewald_params_from_num_k(cells, periodic, NUM_K)
    for index in range(2):
        single = ewald_params_from_num_k(cells[index], periodic[index], NUM_K)
        for value, reference in zip(single, batched, strict=True):
            assert value.shape == ()
            assert_close(value, reference[index])


def test_params_scale_with_the_cell_volume():
    small = ewald_params_from_num_k(
        2.0 * torch.eye(3, dtype=DTYPE), torch.tensor([True] * 3), NUM_K
    )
    large = ewald_params_from_num_k(
        4.0 * torch.eye(3, dtype=DTYPE), torch.tensor([True] * 3), NUM_K
    )
    # lambda ~ V^(1/3), so doubling every cell vector doubles all three parameters
    for value, reference in zip(large, small, strict=True):
        assert_close(value, 2.0 * reference)


def test_params_factors_scale_smearing_and_cutoff():
    cell = 4.0 * torch.eye(3, dtype=DTYPE)
    periodic = torch.tensor([True, True, True])
    # the defaults are the rules of thumb sigma = 2 * lambda and r_c = 5 * sigma
    lam, sigma, cutoff = ewald_params_from_num_k(cell, periodic, NUM_K)
    assert_close(sigma, 2.0 * lam)
    assert_close(cutoff, 5.0 * sigma)

    lam_c, sigma_c, cutoff_c = ewald_params_from_num_k(
        cell, periodic, NUM_K, smearing_factor=3.0, cutoff_factor=7.0
    )
    assert_close(lam_c, lam)  # the k-resolution only depends on num_k
    assert_close(sigma_c, 3.0 * lam)
    assert_close(cutoff_c, 7.0 * sigma_c)


def test_params_realize_roughly_num_k_vectors():
    cell = torch.diag(torch.tensor([4.0, 7.0, 11.0], dtype=DTYPE))
    periodic = torch.tensor([True, True, True])
    lam, _, _ = ewald_params_from_num_k(cell, periodic, NUM_K)
    lengths = torch.linalg.norm(cell, dim=-1)
    ns = [max(1, int(torch.ceil(lengths[i] / lam))) for i in range(3)]
    realized = _integer_kgrid(ns, halfspace=True).shape[0]
    assert NUM_K * 0.5 < realized < NUM_K * 2.0


def test_params_nonperiodic_placeholders():
    cell = torch.zeros(3, 3, dtype=DTYPE)
    lam, sigma, cutoff = ewald_params_from_num_k(
        cell, torch.tensor([False, False, False]), NUM_K
    )
    assert float(lam) == 0.0
    assert float(sigma) == 1.0  # a placeholder, never used: there is no k-space part
    assert float(cutoff) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_k": 0}, "`num_k` must be a positive integer"),
        ({"smearing_factor": 0.0}, "`smearing_factor` must be positive"),
        ({"cutoff_factor": -1.0}, "`cutoff_factor` must be positive"),
    ],
)
def test_params_reject_invalid_arguments(kwargs, match):
    cell = 4.0 * torch.eye(3, dtype=DTYPE)
    periodic = torch.tensor([True, True, True])
    kwargs.setdefault("num_k", NUM_K)
    with pytest.raises(ValueError, match=match):
        ewald_params_from_num_k(cell, periodic, **kwargs)


# --- _integer_kgrid -----------------------------------------------------------------


@pytest.mark.parametrize("ns", [[3, 3, 3], [4, 5, 6], [1, 1, 7], [2, 3, 4]])
def test_kgrid_full_covers_the_box(ns):
    grid = _integer_kgrid(ns, halfspace=False)
    assert grid.shape == (ns[0] * ns[1] * ns[2], 3)
    assert torch.unique(grid, dim=0).shape[0] == grid.shape[0]
    assert bool((grid == 0).all(dim=-1).any())  # k = 0 is part of the full grid


@pytest.mark.parametrize("ns", [[3, 3, 3], [4, 5, 6], [1, 1, 7], [2, 3, 4]])
def test_kgrid_halfspace_keeps_one_of_each_pair(ns):
    grid = _integer_kgrid(ns, halfspace=True)
    assert not bool((grid == 0).all(dim=-1).any())  # k = 0 is excluded
    assert torch.unique(grid, dim=0).shape[0] == grid.shape[0]

    # no vector and its opposite are both kept
    rows = {tuple(int(v) for v in row) for row in grid}
    assert all(tuple(-v for v in row) not in rows for row in rows)

    # the half space, its mirror image and k = 0 reconstruct the full grid: this only
    # holds when the grid is symmetric, i.e. for odd per-axis extents
    if all(n % 2 == 1 for n in ns):
        full = _integer_kgrid(ns, halfspace=False)
        rebuilt = torch.cat([grid, -grid, torch.zeros(1, 3, dtype=grid.dtype)])
        assert_close(
            torch.unique(rebuilt, dim=0).sort(dim=0).values,
            torch.unique(full, dim=0).sort(dim=0).values,
        )


# --- prepare_tiled_batch ------------------------------------------------------------


def test_batch_layout_and_dtypes():
    samples = [crystal(box=4.0), slab(), cluster()]
    batch, tiling = collate(samples)

    n_atoms = [s["positions"].shape[0] for s in samples]
    assert batch["positions"].shape == (sum(n_atoms), 3)
    assert batch["charges"].shape == (sum(n_atoms), 1)
    assert batch["cell"].shape == (len(samples), 3, 3)
    assert batch["periodic"].shape == (len(samples), 3)
    assert batch["periodic"].dtype == torch.bool
    assert batch["positions"].dtype == DTYPE
    assert_close(
        torch.bincount(batch["system_index"], minlength=len(samples)),
        torch.tensor(n_atoms),
    )
    assert tiling["sigma"].shape == (len(samples),)
    assert tiling["sigma"].dtype == DTYPE
    assert tiling["k_int"].dtype == torch.long
    assert tiling["gather_mask"].dtype == torch.bool


def test_batch_orders_screened_pairs_before_bare_ones():
    periodic_samples = [crystal(box=4.0), slab()]
    samples = [periodic_samples[0], cluster(), periodic_samples[1]]
    batch, tiling = collate(samples)

    n_screened = int(tiling["n_screened_pairs"])
    assert n_screened == sum(s["neighbor_indices"].shape[0] for s in periodic_samples)
    assert batch["neighbor_indices"].shape[0] == sum(
        s["neighbor_indices"].shape[0] for s in samples
    )

    # every screened pair carries the smearing of its own system ...
    expected = torch.cat(
        [
            tiling["sigma"][index].expand(s["neighbor_indices"].shape[0])
            for index, s in enumerate(samples)
            if bool(s["periodic"].any())
        ]
    )
    assert_close(tiling["sigma_pair"], expected)
    # ... and the bare pairs, which have no smearing, come last
    bare = batch["neighbor_indices"][n_screened:]
    assert bool(tiling["pbc_atom"][bare.flatten()].logical_not().all())


def test_batch_pads_the_k_axis_to_a_tile_multiple():
    samples = [crystal(box=4.0), crystal(box=6.0, seed=11)]
    _, tiling = collate(samples, block_kvecs=64, k_pad_fraction=1.0)
    k_pad = tiling["k_int"].shape[1]
    assert k_pad % 64 == 0
    assert k_pad >= (1.0 + 1.0) * NUM_K
    assert int(tiling["n_kvec_tiles"]) == k_pad // 64

    # every system is padded, and the padded rows are the trailing zero vectors, which
    # the potentials map to a zero kernel
    realized = (tiling["k_int"].abs().sum(dim=-1) > 0).sum(dim=-1)
    assert bool((realized < k_pad).all())
    for row, count in enumerate(realized):
        assert bool((tiling["k_int"][row, count:] == 0).all())


def test_batch_flat_layout_covers_every_periodic_atom_once_per_k_tile():
    samples = [crystal(box=4.0), cluster(), slab()]
    _, tiling = collate(samples, block_atoms=8)

    n_periodic_atoms = sum(
        s["positions"].shape[0] for s in samples if bool(s["periodic"].any())
    )
    assert int(tiling["flat_mask"].sum()) == n_periodic_atoms

    gathered = tiling["atom_gather"][tiling["gather_mask"]]
    counts = torch.bincount(gathered, minlength=tiling["pbc_atom"].shape[0])
    n_kvec_tiles = int(tiling["n_kvec_tiles"])
    assert bool((counts[tiling["pbc_atom"]] == n_kvec_tiles).all())
    assert bool((counts[~tiling["pbc_atom"]] == 0).all())


def test_batch_dispatch_table_groups_atom_tiles_contiguously():
    # the kernel's second pass reduces with a reshape instead of a scatter, which
    # requires each (system, atom_tile) group to be contiguous in the dispatch table
    samples = [crystal(box=4.0, n_atoms=20), crystal(box=6.0, n_atoms=9, seed=11)]
    _, tiling = collate(samples, block_atoms=8, block_kvecs=128)

    b_col, kt_col = tiling["b_col"], tiling["kt_col"]
    n_kvec_tiles = int(tiling["n_kvec_tiles"])
    assert bool((b_col.diff() >= 0).all())
    assert b_col.shape[0] % n_kvec_tiles == 0
    # within each group the k-tile index runs 0, 1, ..., n_kvec_tiles - 1
    assert_close(
        kt_col.reshape(-1, n_kvec_tiles),
        torch.arange(n_kvec_tiles).expand(b_col.shape[0] // n_kvec_tiles, -1),
    )
    # and the group's atoms are all the same
    grouped = tiling["atom_gather"].reshape(
        -1, n_kvec_tiles, tiling["atom_gather"].shape[-1]
    )
    assert bool((grouped == grouped[:, :1]).all())


def test_batch_pbc_atom_marks_periodic_systems():
    samples = [crystal(box=4.0), cluster(), slab()]
    _, tiling = collate(samples)
    expected = torch.cat(
        [
            torch.full(
                (s["positions"].shape[0],), bool(s["periodic"].any()), dtype=torch.bool
            )
            for s in samples
        ]
    )
    assert torch.equal(tiling["pbc_atom"], expected)
    assert torch.equal(
        tiling["pbc_system"], torch.tensor([0, 2])
    )  # the two periodic rows


@pytest.mark.parametrize(("halfspace", "g_factor"), [(True, 2.0), (False, 1.0)])
def test_batch_g_factor_follows_halfspace(halfspace, g_factor):
    _, tiling = collate([crystal(box=4.0)], halfspace=halfspace)
    assert float(tiling["g_factor"]) == g_factor


def test_batch_smearing_factor_scales_the_smearing():
    samples = [crystal(box=4.0), crystal(box=6.0, seed=11)]
    _, default = collate(samples)
    _, scaled = collate(samples, smearing_factor=3.0)
    assert_close(scaled["sigma"], 1.5 * default["sigma"])
    # the k-grids are sized by num_k alone, so their shape does not change
    assert scaled["k_int"].shape == default["k_int"].shape


@pytest.mark.parametrize(
    "override", [torch.tensor(0.7, dtype=DTYPE), torch.tensor([0.7, 0.7], dtype=DTYPE)]
)
def test_batch_smearing_override_skips_nonperiodic_systems(override):
    samples = [crystal(box=4.0), cluster()]
    _, default = collate(samples)
    _, tiling = collate(samples, smearing=override)
    assert float(tiling["sigma"][0]) == 0.7
    assert float(tiling["sigma"][1]) == float(default["sigma"][1])


@pytest.mark.parametrize("device", DEVICES)
def test_batch_stays_on_the_input_device(device):
    samples = [crystal(box=4.0), cluster(), slab()]
    samples = [{key: value.to(device) for key, value in s.items()} for s in samples]

    batch, tiling = collate(samples)
    expected = torch.empty(0, device=device).device
    for name, value in list(batch.items()) + list(tiling.items()):
        assert value.device == expected, f"{name} landed on {value.device}"


def test_batch_warns_when_the_k_padding_window_is_exceeded():
    with pytest.warns(UserWarning, match="above the padding window"):
        collate([crystal(box=4.0)], k_pad_fraction=0.0, num_k=10)


def test_batch_tiling_carries_no_gradient():
    """The tiling is static data: it must not tie the graph to cells or positions."""
    samples = [crystal(box=4.0), slab(), cluster()]
    for sample in samples:
        sample["cell"] = sample["cell"].clone().requires_grad_(True)
        sample["positions"] = sample["positions"].clone().requires_grad_(True)

    batch, tiling = collate(samples)
    for name, value in tiling.items():
        assert not value.requires_grad, f"{name} carries a gradient"
    # while the batch is made of the caller's own tensors
    assert batch["cell"].requires_grad
    assert batch["positions"].requires_grad


# --- TorchScript --------------------------------------------------------------------


@pytest.mark.parametrize(
    "function", [shrink_2d_cell, ewald_params_from_num_k, prepare_tiled_batch]
)
def test_functions_are_scriptable(function):
    assert torch.jit.script(function) is not None


def test_scripted_collation_matches_eager():
    samples = [crystal(box=4.0), cluster(), slab(), crystal(box=6.0, seed=11)]
    kwargs = {
        "positions": [s["positions"] for s in samples],
        "charges": [s["charges"] for s in samples],
        "cells": [s["cell"] for s in samples],
        "periodic": [s["periodic"] for s in samples],
        "neighbor_indices": [s["neighbor_indices"] for s in samples],
        "neighbor_distances": [s["neighbor_distances"] for s in samples],
        "num_k": NUM_K,
    }
    eager_batch, eager_tiling = prepare_tiled_batch(**kwargs)
    scripted_batch, scripted_tiling = torch.jit.script(prepare_tiled_batch)(**kwargs)

    for key, value in eager_batch.items():
        assert torch.equal(value, scripted_batch[key]), f"batch entry {key} differs"
    for key, value in eager_tiling.items():
        assert torch.equal(value, scripted_tiling[key]), f"tiling entry {key} differs"


# --- error paths --------------------------------------------------------------------


def test_reject_empty_batch():
    with pytest.raises(ValueError, match="Cannot collate an empty list"):
        collate([])


def test_reject_mismatched_list_lengths():
    sample = crystal(box=4.0)
    with pytest.raises(ValueError, match="same length"):
        prepare_tiled_batch(
            positions=[sample["positions"], sample["positions"]],
            charges=[sample["charges"]],
            cells=[sample["cell"]],
            periodic=[sample["periodic"]],
            neighbor_indices=[sample["neighbor_indices"]],
            neighbor_distances=[sample["neighbor_distances"]],
            num_k=NUM_K,
        )


@pytest.mark.parametrize(("block_atoms", "block_kvecs"), [(0, 128), (32, 0), (-1, -1)])
def test_reject_nonpositive_block_sizes(block_atoms, block_kvecs):
    with pytest.raises(ValueError, match="must be positive"):
        collate([crystal(box=4.0)], block_atoms=block_atoms, block_kvecs=block_kvecs)


def test_reject_1d_periodicity():
    sample = crystal(box=4.0, periodic=(True, False, False))
    with pytest.raises(ValueError, match="1D-periodic"):
        collate([sample])


def test_reject_mismatched_channel_counts():
    samples = [crystal(box=4.0), crystal(box=6.0, seed=11)]
    samples[1]["charges"] = torch.cat([samples[1]["charges"]] * 2, dim=1)
    with pytest.raises(ValueError, match="same number of charge channels"):
        collate(samples)


def test_reject_mismatched_dtypes():
    samples = [crystal(box=4.0), crystal(box=6.0, seed=11, dtype=torch.float32)]
    with pytest.raises(ValueError, match="dtype and device of the first system"):
        collate(samples)


def test_reject_inconsistent_single_system():
    # the per-system checks are delegated to the same helper the calculators use
    sample = crystal(box=4.0)
    sample["charges"] = sample["charges"].squeeze(-1)
    with pytest.raises(ValueError, match="`charges` must be a 2-dimensional tensor"):
        collate([sample])


def test_reject_non_bool_periodic():
    sample = crystal(box=4.0)
    sample["periodic"] = torch.ones(3, dtype=torch.int64)
    with pytest.raises(TypeError, match="must be torch.bool"):
        collate([sample])
