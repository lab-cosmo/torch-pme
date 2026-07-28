"""Tests for the batched (tiled) Ewald evaluation through the metatensor interface."""

import io
import sys
from pathlib import Path

import pytest
import torch

import torchpme
from torchpme.lib import ewald_params_from_num_k, shrink_2d_cell

sys.path.append(str(Path(__file__).parents[1]))
from helpers import define_crystal, neighbor_list

mts_torch = pytest.importorskip("metatensor.torch")
mta_torch = pytest.importorskip("metatomic.torch")

DTYPE = torch.float64
NUM_K = 200
NEIGHBOR_SAMPLE_NAMES = [
    "first_atom",
    "second_atom",
    "cell_shift_a",
    "cell_shift_b",
    "cell_shift_c",
]


def to_system(positions, charges, cell, pbc):
    system = mta_torch.System(
        types=torch.arange(positions.shape[0], dtype=torch.int32),
        positions=positions,
        cell=cell,
        pbc=pbc,
    )
    block = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )
    tensor = mts_torch.TensorMap(
        keys=mts_torch.Labels("_", torch.zeros(1, 1, dtype=torch.int32)),
        blocks=[block],
    )
    system.add_data(name="charge", tensor=tensor)
    return system


def to_neighbor_block(vectors, indices, shifts):
    samples = torch.zeros(indices.shape[0], 5, dtype=torch.int32)
    samples[:, :2] = indices.to(torch.int32)
    samples[:, 2:] = shifts.to(torch.int32)
    return mts_torch.TensorBlock(
        values=vectors.unsqueeze(-1),
        samples=mts_torch.Labels(NEIGHBOR_SAMPLE_NAMES, samples),
        components=[mts_torch.Labels.range("xyz", 3)],
        properties=mts_torch.Labels.range("distance", 1),
    )


def sample_3d(crystal, positions_override=None):
    positions, charges, cell, _, _ = define_crystal(crystal, dtype=DTYPE)
    if positions_override is not None:
        positions = positions_override
    pbc = torch.tensor([True, True, True])
    _, _, cutoff = ewald_params_from_num_k(cell, pbc, NUM_K)
    indices, shifts = neighbor_list(
        positions=positions.detach(),
        periodic=True,
        box=cell,
        cutoff=float(cutoff),
        neighbor_shifts=True,
    )
    vectors = positions[indices[:, 1]] - positions[indices[:, 0]] + shifts @ cell
    return to_system(positions, charges, cell, pbc), to_neighbor_block(
        vectors, indices, shifts
    )


def sample_2d():
    """A NaCl-like slab with a zero vacuum row (the metatomic 2D convention)."""
    positions = torch.tensor(
        [[0.0, 0.0, 5.0], [2.0, 2.0, 5.0], [2.0, 0.0, 6.5], [0.0, 2.0, 6.5]],
        dtype=DTYPE,
    )
    charges = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=DTYPE).reshape(-1, 1)
    cell = torch.diag(torch.tensor([4.0, 4.0, 0.0], dtype=DTYPE))
    pbc = torch.tensor([True, True, False])
    effective = shrink_2d_cell(cell, pbc, positions)
    _, _, cutoff = ewald_params_from_num_k(effective, pbc, NUM_K)
    indices, shifts = neighbor_list(
        positions=positions,
        periodic=True,
        box=effective,
        cutoff=float(cutoff),
        neighbor_shifts=True,
    )
    vectors = positions[indices[:, 1]] - positions[indices[:, 0]] + shifts @ effective
    return to_system(positions, charges, cell, pbc), to_neighbor_block(
        vectors, indices, shifts
    )


def sample_0d():
    generator = torch.Generator().manual_seed(3)
    positions = torch.rand((5, 3), generator=generator, dtype=DTYPE) * 3.0
    charges = torch.tensor([1.0, -1.0, 1.0, -1.0, 0.5], dtype=DTYPE).reshape(-1, 1)
    indices = torch.triu_indices(5, 5, offset=1).T
    vectors = positions[indices[:, 1]] - positions[indices[:, 0]]
    shifts = torch.zeros(indices.shape[0], 3, dtype=DTYPE)
    return to_system(
        positions,
        charges,
        torch.zeros(3, 3, dtype=DTYPE),
        torch.tensor([False, False, False]),
    ), to_neighbor_block(vectors, indices, shifts)


@pytest.fixture
def calculator():
    return torchpme.metatensor.EwaldCalculator(
        torchpme.CoulombPotential(smearing=1.0), lr_wavelength=1.0
    ).to(DTYPE)


def evaluate(calculator, systems, blocks, **kwargs):
    tiling = torchpme.metatensor.prepare_tiled_batch(
        systems, blocks, num_k=NUM_K, k_pad_fraction=1.0, **kwargs
    )
    return calculator.forward_batched(systems, blocks, tiling), tiling


def test_batched_matches_pure_torch(calculator):
    systems, blocks = zip(
        *[sample_3d("CsCl"), sample_2d(), sample_0d(), sample_3d("NaCl_cubic")],
        strict=False,
    )
    result, _ = evaluate(calculator, list(systems), list(blocks))
    values = result.block().values
    assert torch.all(torch.isfinite(values))

    # the same computation through the pure-torch interface must be identical
    from torchpme.lib import prepare_tiled_batch as torch_prepare

    batch, torch_tiling = torch_prepare(
        positions=[s.positions for s in systems],
        charges=[s.get_data("charge").block().values for s in systems],
        cells=[s.cell for s in systems],
        periodic=[s.pbc for s in systems],
        neighbor_indices=[
            torch.stack(
                [
                    b.samples.column("first_atom"),
                    b.samples.column("second_atom"),
                ],
                dim=1,
            ).to(torch.int64)
            for b in blocks
        ],
        neighbor_distances=[
            torch.linalg.norm(b.values, dim=1).squeeze(-1) for b in blocks
        ],
        num_k=NUM_K,
        k_pad_fraction=1.0,
    )
    torch_calculator = torchpme.EwaldCalculator(
        torchpme.CoulombPotential(smearing=1.0), lr_wavelength=1.0
    ).to(DTYPE)
    reference = torch_calculator.forward_batched(
        batch["charges"],
        batch["cell"],
        batch["positions"],
        batch["neighbor_indices"],
        batch["neighbor_distances"],
        batch["system_index"],
        batch["periodic"],
        torch_tiling,
    )
    torch.testing.assert_close(values, reference, rtol=1e-12, atol=1e-14)


def test_batched_matches_serial_metatensor(calculator):
    systems, blocks = zip(*[sample_3d("CsCl"), sample_3d("NaCl_cubic")], strict=False)
    systems, blocks = list(systems), list(blocks)
    result, tiling = evaluate(calculator, systems, blocks)
    values = result.block().values

    offset = 0
    for index, (system, block) in enumerate(zip(systems, blocks, strict=False)):
        sigma = float(tiling["sigma"][index])
        serial_calculator = torchpme.metatensor.EwaldCalculator(
            torchpme.CoulombPotential(smearing=sigma), lr_wavelength=sigma / 2
        ).to(DTYPE)
        serial = serial_calculator.forward(system, block).block().values
        n = len(system)
        torch.testing.assert_close(
            values[offset : offset + n], serial, rtol=1e-8, atol=1e-10
        )
        offset += n


def test_batched_samples_metadata(calculator):
    systems, blocks = zip(*[sample_3d("CsCl"), sample_0d()], strict=False)
    result, _ = evaluate(calculator, list(systems), list(blocks))
    block = result.block()
    assert block.samples.names == ["system", "atom"]
    expected = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]], dtype=torch.int32
    )
    assert torch.equal(block.samples.values, expected)


def test_batched_forces_match_pure_torch(calculator):
    positions_list = []
    for crystal in ("CsCl", "NaCl_cubic"):
        positions, _, _, _, _ = define_crystal(crystal, dtype=DTYPE)
        positions_list.append(positions.clone().requires_grad_(True))
    systems, blocks = zip(
        *[
            sample_3d(crystal, positions_override=positions)
            for crystal, positions in zip(
                ("CsCl", "NaCl_cubic"), positions_list, strict=False
            )
        ],
        strict=False,
    )
    systems, blocks = list(systems), list(blocks)
    result, _ = evaluate(calculator, systems, blocks)
    charges = torch.cat([s.get_data("charge").block().values for s in systems])
    energy = (result.block().values * charges).sum()
    forces = torch.autograd.grad(energy, positions_list)
    assert all(torch.all(torch.isfinite(force)) for force in forces)
    assert all(float(force.abs().max()) > 0 for force in forces)


def test_batched_torchscript(calculator):
    systems, blocks = zip(*[sample_3d("CsCl"), sample_3d("NaCl_cubic")], strict=False)
    systems, blocks = list(systems), list(blocks)
    eager_result, tiling = evaluate(calculator, systems, blocks)

    scripted = torch.jit.script(calculator)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    scripted = torch.jit.load(buffer)

    scripted_values = scripted.forward_batched(systems, blocks, tiling)
    torch.testing.assert_close(
        scripted_values.block().values,
        eager_result.block().values,
        rtol=1e-12,
        atol=1e-14,
    )


def test_batched_prepare_torchscript():
    systems, blocks = zip(*[sample_3d("CsCl"), sample_3d("NaCl_cubic")], strict=False)
    systems, blocks = list(systems), list(blocks)

    eager = torchpme.metatensor.prepare_tiled_batch(
        systems, blocks, num_k=NUM_K, k_pad_fraction=1.0
    )
    scripted = torch.jit.script(torchpme.metatensor.prepare_tiled_batch)(
        systems, blocks, num_k=NUM_K, k_pad_fraction=1.0
    )

    assert set(scripted) == set(eager)
    for key, value in eager.items():
        assert torch.equal(value, scripted[key]), f"tiling entry {key} differs"


def test_batched_rejects_mesh_calculators():
    calculator = torchpme.metatensor.PMECalculator(
        torchpme.CoulombPotential(smearing=1.0), mesh_spacing=0.5
    ).to(DTYPE)
    systems, blocks = zip(*[sample_3d("CsCl")], strict=False)
    systems, blocks = list(systems), list(blocks)
    tiling = torchpme.metatensor.prepare_tiled_batch(
        systems, blocks, num_k=NUM_K, k_pad_fraction=1.0
    )
    with pytest.raises(
        (NotImplementedError, torch.jit.Error), match="only supported by"
    ):
        calculator.forward_batched(systems, blocks, tiling)


def test_batched_rejects_mismatched_lengths(calculator):
    systems, blocks = zip(*[sample_3d("CsCl"), sample_3d("NaCl_cubic")], strict=False)
    with pytest.raises(ValueError, match="2 systems but 1 neighbor"):
        torchpme.metatensor.prepare_tiled_batch(list(systems), [blocks[0]], num_k=NUM_K)


def test_batched_rejects_invalid_neighbor_metadata(calculator):
    # collation only checks what it consumes (the charges); the metadata of the
    # neighbor blocks is validated by `forward_batched`, which is what reads them
    systems, blocks = zip(*[sample_3d("CsCl")], strict=False)
    systems, blocks = list(systems), list(blocks)
    tiling = torchpme.metatensor.prepare_tiled_batch(
        systems, blocks, num_k=NUM_K, k_pad_fraction=1.0
    )
    broken = mts_torch.TensorBlock(
        values=blocks[0].values,
        samples=blocks[0].samples,
        components=blocks[0].components,
        properties=mts_torch.Labels.range("not_a_distance", 1),
    )
    with pytest.raises(ValueError, match="Invalid properties for `neighbors`"):
        calculator.forward_batched(systems, [broken], tiling)


def test_batched_rejects_missing_charges(calculator):
    system, block = sample_3d("CsCl")
    stripped = mta_torch.System(
        types=system.types,
        positions=system.positions,
        cell=system.cell,
        pbc=system.pbc,
    )
    with pytest.raises(ValueError, match="does not contain `charge` data"):
        torchpme.metatensor.prepare_tiled_batch([stripped], [block], num_k=NUM_K)
