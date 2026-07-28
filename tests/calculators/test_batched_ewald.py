"""
Tests for the batched (tiled) Ewald evaluation:
:meth:`EwaldCalculator.forward_batched` fed by :func:`torchpme.lib.prepare_tiled_batch`.

A tiled batch must reproduce the per-structure serial evaluation for 3D crystals, 2D
slabs (via the slab correction; exactly when the vacuum-shrink is a no-op, and up to
the shrink residual otherwise) and non-periodic clusters (bare potential). The result
must be independent of the tile sizes, differentiable (forces), and TorchScriptable.

Realistic structures come from the two MAD subsets in ``tests/resources`` (solvated
ions in 3D cells and catalytic surfaces as 2D slabs); the small synthetic samples
below cover what those frames cannot: a non-periodic cluster, the branch where the
2D vacuum-shrink is a no-op, and the structural error paths.
"""

import io
import sys
from pathlib import Path

import pytest
import torch

from torchpme import (
    Calculator,
    CoulombPotential,
    EwaldCalculator,
    InversePowerLawPotential,
    PMECalculator,
)
from torchpme.lib import ewald_params_from_num_k, prepare_tiled_batch, shrink_2d_cell

sys.path.append(str(Path(__file__).parents[1]))
from helpers import (
    MAD_CAT_FRAMES,
    MAD_SOL_FRAMES,
    define_crystal,
    mad_sample,
    neighbor_list,
)

DTYPE = torch.float64
NUM_K = 200
MAD_FRAMES = 4  # how many frames of each MAD subset to run


def sample_3d(crystal):
    positions, charges, cell, _, _ = define_crystal(crystal, dtype=DTYPE)
    periodic = torch.tensor([True, True, True])
    _, _, cutoff = ewald_params_from_num_k(cell, periodic, NUM_K)
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=positions, periodic=True, box=cell, cutoff=float(cutoff)
    )
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "periodic": periodic,
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": neighbor_distances,
    }


def sample_2d(vacuum):
    """
    A small NaCl-like slab; for ``vacuum <= 7.5`` the cell-shrink is a no-op.

    Complements the ``mad_cat`` frames, whose vacuum is always shrunk, so the branch
    where the effective cell equals the true one is covered too.
    """
    positions = torch.tensor(
        [[0.0, 0.0, 5.0], [2.0, 2.0, 5.0], [2.0, 0.0, 6.5], [0.0, 2.0, 6.5]],
        dtype=DTYPE,
    )
    charges = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=DTYPE).reshape(-1, 1)
    cell = torch.diag(torch.tensor([4.0, 4.0, vacuum], dtype=DTYPE))
    periodic = torch.tensor([True, True, False])
    effective = shrink_2d_cell(cell, periodic, positions)
    _, _, cutoff = ewald_params_from_num_k(effective, periodic, NUM_K)
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=positions, periodic=True, box=cell, cutoff=float(cutoff)
    )
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "periodic": periodic,
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": neighbor_distances,
    }


def sample_0d():
    """A small charged cluster; the pair list contains all pairs (half list)."""
    generator = torch.Generator().manual_seed(3)
    positions = torch.rand((5, 3), generator=generator, dtype=DTYPE) * 3.0
    charges = torch.tensor([1.0, -1.0, 1.0, -1.0, 0.5], dtype=DTYPE).reshape(-1, 1)
    neighbor_indices = torch.triu_indices(5, 5, offset=1).T
    neighbor_distances = (
        positions[neighbor_indices[:, 0]] - positions[neighbor_indices[:, 1]]
    ).norm(dim=-1)
    return {
        "positions": positions,
        "charges": charges,
        "cell": torch.zeros(3, 3, dtype=DTYPE),
        "periodic": torch.tensor([False, False, False]),
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": neighbor_distances,
    }


def collate(samples, **kwargs):
    # for the tiny test cells the per-axis rounding overshoots the default padding
    # window, so use a wide one (the window only affects shape stability, not values)
    kwargs.setdefault("k_pad_fraction", 1.0)
    return prepare_tiled_batch(
        positions=[s["positions"] for s in samples],
        charges=[s["charges"] for s in samples],
        cells=[s["cell"] for s in samples],
        periodic=[s["periodic"] for s in samples],
        neighbor_indices=[s["neighbor_indices"] for s in samples],
        neighbor_distances=[s["neighbor_distances"] for s in samples],
        num_k=NUM_K,
        **kwargs,
    )


def tiled_potential(calculator, batch, tiling):
    return calculator.forward_batched(
        batch["charges"],
        batch["cell"],
        batch["positions"],
        batch["neighbor_indices"],
        batch["neighbor_distances"],
        batch["system_index"],
        batch["periodic"],
        tiling,
    )


def serial_potential(sample, sigma, make_potential):
    """Serial reference with the same per-system smearing and k-resolution."""
    if not bool(sample["periodic"].any()):
        calculator = Calculator(make_potential(None)).to(DTYPE)
        cell = torch.eye(3, dtype=DTYPE)
    else:
        # lr_wavelength = lambda_b = sigma_b / 2 reproduces the tiled per-axis extents
        calculator = EwaldCalculator(make_potential(sigma), lr_wavelength=sigma / 2).to(
            DTYPE
        )
        cell = sample["cell"]
    return calculator.forward(
        sample["charges"],
        cell,
        sample["positions"],
        sample["neighbor_indices"],
        sample["neighbor_distances"],
        periodic=sample["periodic"],
    )


def assert_matches_serial(calculator, samples, make_potential, rtol=1e-8, atol=1e-10):
    batch, tiling = collate(samples)
    tiled = tiled_potential(calculator, batch, tiling)
    assert torch.all(torch.isfinite(tiled))
    offset = 0
    for index, sample in enumerate(samples):
        n = sample["positions"].shape[0]
        reference = serial_potential(
            sample, float(tiling["sigma"][index]), make_potential
        )
        torch.testing.assert_close(
            tiled[offset : offset + n], reference, rtol=rtol, atol=atol
        )
        offset += n


def coulomb(smearing):
    return CoulombPotential(smearing=smearing)


@pytest.fixture
def calculator():
    # the constructor smearing / lr_wavelength are overridden per system by the tiling
    return EwaldCalculator(CoulombPotential(smearing=1.0), lr_wavelength=1.0).to(DTYPE)


def test_tiled_3d_vs_serial(calculator):
    samples = [sample_3d("CsCl"), sample_3d("NaCl_cubic"), sample_3d("zincblende")]
    assert_matches_serial(calculator, samples, coulomb)


def test_tiled_single_system_vs_serial(calculator):
    assert_matches_serial(calculator, [sample_3d("CsCl")], coulomb)


def test_tiled_batch_order_independence(calculator):
    samples = [sample_3d("CsCl"), sample_2d(vacuum=7.0), sample_0d()]
    batch, tiling = collate(samples)
    values = tiled_potential(calculator, batch, tiling)

    reordered = [samples[2], samples[0], samples[1]]
    batch_r, tiling_r = collate(reordered)
    values_r = tiled_potential(calculator, batch_r, tiling_r)

    sizes = [s["positions"].shape[0] for s in samples]
    split = torch.split(values, sizes)
    split_r = torch.split(values_r, [sizes[2], sizes[0], sizes[1]])
    torch.testing.assert_close(split[0], split_r[1], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(split[1], split_r[2], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(split[2], split_r[0], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("block_atoms", "block_kvecs"), [(8, 16), (1, 1), (64, 512)])
def test_tiled_tile_size_invariance(calculator, block_atoms, block_kvecs):
    samples = [sample_3d("CsCl"), sample_2d(vacuum=7.0), sample_0d()]
    batch, tiling = collate(samples)
    reference = tiled_potential(calculator, batch, tiling)
    _, tiling_other = collate(samples, block_atoms=block_atoms, block_kvecs=block_kvecs)
    other = tiled_potential(calculator, batch, tiling_other)
    torch.testing.assert_close(other, reference, rtol=1e-12, atol=1e-14)


def test_tiled_mixed_periodicity_vs_serial(calculator):
    # vacuum=7.0 keeps the 2D cell-shrink a no-op, so all references are tight
    samples = [
        sample_3d("CsCl"),
        sample_2d(vacuum=7.0),
        sample_0d(),
        sample_3d("NaCl_cubic"),
    ]
    assert_matches_serial(calculator, samples, coulomb)


def test_tiled_2d_shrunk_slab_vs_serial(calculator):
    # with a large vacuum the tiled path evaluates the shrunk effective cell; it
    # agrees with the serial (unshrunk) evaluation up to the shrink residual ~e^{-3pi}
    samples = [sample_2d(vacuum=20.0)]
    assert_matches_serial(calculator, samples, coulomb, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("frame_index", range(MAD_FRAMES))
def test_tiled_mad_solvated_vs_serial(calculator, frame_index):
    # solvated ions: 3D-periodic cells of ~140 atoms
    samples = [mad_sample(MAD_SOL_FRAMES, frame_index, NUM_K, dtype=DTYPE)]
    assert_matches_serial(calculator, samples, coulomb)


@pytest.mark.parametrize("frame_index", range(MAD_FRAMES))
def test_tiled_mad_surfaces_vs_serial(calculator, frame_index):
    # catalytic surfaces: 2D slabs with ~100 A of vacuum along a non-orthogonal axis,
    # so the tiled path evaluates a shrunk effective cell and agrees with the serial
    # (unshrunk) evaluation only up to the shrink residual ~e^{-3pi}
    samples = [mad_sample(MAD_CAT_FRAMES, frame_index, NUM_K, dtype=DTYPE)]
    assert_matches_serial(calculator, samples, coulomb, rtol=1e-3, atol=1e-4)


def test_tiled_mad_mixed_batch_vs_serial(calculator):
    # one batch spanning both subsets: different sizes, cells and periodicities
    samples = [
        mad_sample(MAD_SOL_FRAMES, 0, NUM_K, dtype=DTYPE),
        mad_sample(MAD_CAT_FRAMES, 0, NUM_K, dtype=DTYPE),
        mad_sample(MAD_SOL_FRAMES, 1, NUM_K, dtype=DTYPE),
        mad_sample(MAD_CAT_FRAMES, 1, NUM_K, dtype=DTYPE),
    ]
    assert_matches_serial(calculator, samples, coulomb, rtol=1e-3, atol=1e-4)


def test_tiled_mad_tile_size_invariance(calculator):
    samples = [
        mad_sample(MAD_SOL_FRAMES, 0, NUM_K, dtype=DTYPE),
        mad_sample(MAD_CAT_FRAMES, 0, NUM_K, dtype=DTYPE),
    ]
    batch, tiling = collate(samples)
    reference = tiled_potential(calculator, batch, tiling)
    _, tiling_other = collate(samples, block_atoms=16, block_kvecs=64)
    other = tiled_potential(calculator, batch, tiling_other)
    torch.testing.assert_close(other, reference, rtol=1e-10, atol=1e-12)


def test_tiled_inverse_power_law_vs_serial(calculator):
    def ipl(smearing):
        return InversePowerLawPotential(exponent=1, smearing=smearing)

    calculator = EwaldCalculator(ipl(1.0), lr_wavelength=1.0).to(DTYPE)
    samples = [sample_3d("CsCl"), sample_3d("NaCl_cubic")]
    assert_matches_serial(calculator, samples, ipl)


def test_tiled_ipl_2d_returns_nan(calculator):
    # the 2D slab correction is only implemented for 1/r; other potentials must fail
    # loudly with NaN on the 2D system's atoms (and stay finite elsewhere)
    calculator = EwaldCalculator(
        InversePowerLawPotential(exponent=2, smearing=1.0), lr_wavelength=1.0
    ).to(DTYPE)
    samples = [sample_3d("CsCl"), sample_2d(vacuum=7.0)]
    batch, tiling = collate(samples)
    values = tiled_potential(calculator, batch, tiling)
    n_3d = samples[0]["positions"].shape[0]
    assert torch.all(torch.isfinite(values[:n_3d]))
    assert torch.all(torch.isnan(values[n_3d:]))


def test_tiled_multichannel_vs_serial(calculator):
    samples = []
    for crystal in ("CsCl", "NaCl_cubic"):
        sample = sample_3d(crystal)
        sample["charges"] = torch.cat(
            [sample["charges"], -0.5 * sample["charges"]], dim=1
        )
        samples.append(sample)
    batch, tiling = collate(samples)
    tiled = tiled_potential(calculator, batch, tiling)
    assert tiled.shape[1] == 2
    assert_matches_serial(calculator, samples, coulomb)


def test_tiled_per_system_smearing(calculator):
    # differently sized cells get different smearings from the same num_k
    samples = [sample_3d("CsCl"), sample_3d("NaCl_cubic")]
    _, tiling = collate(samples)
    assert float(tiling["sigma"][0]) != float(tiling["sigma"][1])


def test_tiled_forces_match_serial(calculator):
    samples = [sample_3d("CsCl"), sample_3d("NaCl_cubic")]
    batch, tiling = collate(samples)

    def in_graph_distances(positions, sample, offset):
        indices = sample["neighbor_indices"]
        _, shifts = neighbor_list(
            positions=sample["positions"],
            periodic=True,
            box=sample["cell"],
            cutoff=float(
                ewald_params_from_num_k(sample["cell"], sample["periodic"], NUM_K)[2]
            ),
            neighbor_shifts=True,
        )
        vectors = (
            positions[offset + indices[:, 0]]
            - positions[offset + indices[:, 1]]
            - shifts @ sample["cell"]
        )
        return vectors.norm(dim=-1)

    # serial forces
    serial_forces = []
    for index, sample in enumerate(samples):
        positions = sample["positions"].clone().requires_grad_(True)
        distances = in_graph_distances(positions, sample, 0)
        sigma = float(tiling["sigma"][index])
        serial_calc = EwaldCalculator(
            CoulombPotential(smearing=sigma), lr_wavelength=sigma / 2
        ).to(DTYPE)
        potential = serial_calc.forward(
            sample["charges"],
            sample["cell"],
            positions,
            sample["neighbor_indices"],
            distances,
        )
        energy = (potential * sample["charges"]).sum()
        (force,) = torch.autograd.grad(energy, positions)
        serial_forces.append(force)
    serial_forces = torch.cat(serial_forces)

    # tiled forces with distances recomputed in-graph from the batched positions
    positions = batch["positions"].clone().requires_grad_(True)
    offset = 0
    distances = []
    for sample in samples:
        distances.append(in_graph_distances(positions, sample, offset))
        offset += sample["positions"].shape[0]
    distances = torch.cat(distances)

    potential = calculator.forward_batched(
        batch["charges"],
        batch["cell"],
        positions,
        batch["neighbor_indices"],
        distances,
        batch["system_index"],
        batch["periodic"],
        tiling,
    )
    energy = (potential * batch["charges"]).sum()
    (tiled_forces,) = torch.autograd.grad(energy, positions)

    torch.testing.assert_close(tiled_forces, serial_forces, rtol=1e-8, atol=1e-10)


def test_tiled_cell_gradient_is_finite(calculator):
    samples = [sample_3d("CsCl"), sample_2d(vacuum=7.0), sample_0d()]
    batch, tiling = collate(samples)
    cell = batch["cell"].clone().requires_grad_(True)
    potential = calculator.forward_batched(
        batch["charges"],
        cell,
        batch["positions"],
        batch["neighbor_indices"],
        batch["neighbor_distances"],
        batch["system_index"],
        batch["periodic"],
        tiling,
    )
    energy = (potential * batch["charges"]).sum()
    (cell_grad,) = torch.autograd.grad(energy, cell)
    assert torch.all(torch.isfinite(cell_grad))


def test_tiled_torchscript(calculator):
    samples = [sample_3d("CsCl"), sample_3d("NaCl_cubic")]
    batch, tiling = collate(samples)
    eager = tiled_potential(calculator, batch, tiling)

    scripted = torch.jit.script(calculator)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    scripted = torch.jit.load(buffer)

    scripted_values = tiled_potential(scripted, batch, tiling)
    torch.testing.assert_close(scripted_values, eager, rtol=1e-12, atol=1e-14)


def test_mesh_calculators_reject_batched():
    calculator = PMECalculator(CoulombPotential(smearing=1.0), mesh_spacing=0.5).to(
        DTYPE
    )
    samples = [sample_3d("CsCl")]
    batch, tiling = collate(samples)
    with pytest.raises(
        (NotImplementedError, torch.jit.Error), match="only supported by"
    ):
        tiled_potential(calculator, batch, tiling)


def test_potential_smearing_override_matches_buffer():
    dist = torch.linspace(0.5, 3.0, 8, dtype=DTYPE)
    k_sq = torch.linspace(0.0, 5.0, 8, dtype=DTYPE)
    override = torch.tensor(0.7, dtype=DTYPE)
    with_buffer = CoulombPotential(smearing=0.7)
    other = CoulombPotential(smearing=2.0)
    torch.testing.assert_close(
        other.sr_from_dist(dist, smearing=override), with_buffer.sr_from_dist(dist)
    )
    torch.testing.assert_close(
        other.lr_from_k_sq(k_sq, smearing=override), with_buffer.lr_from_k_sq(k_sq)
    )
    torch.testing.assert_close(
        other.self_contribution(smearing=override), with_buffer.self_contribution()
    )
    torch.testing.assert_close(
        other.background_correction(smearing=override),
        with_buffer.background_correction(),
    )


def test_reject_1d_periodicity(calculator):
    sample = sample_3d("CsCl")
    sample["periodic"] = torch.tensor([True, False, False])
    with pytest.raises(ValueError, match="1D-periodic"):
        collate([sample])


def test_k_pad_window_warning():
    # a cell whose per-axis rounding overshoots the padding window warns that the
    # padded k-count (and thus the batch shape) grows beyond the num_k window
    with pytest.warns(UserWarning, match="above the padding window"):
        collate([sample_2d(vacuum=7.0)], k_pad_fraction=0.1)
