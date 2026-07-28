r"""
Cross-implementation check of the batched (tiled) Ewald against `jax-pme`.

:meth:`EwaldCalculator.forward_batched` and ``jaxpme.batched_tiled`` are two independent
implementations of the same tiled batching scheme. The rest of the suite validates the
tiled path against torch-pme's *own* serial evaluation, which cannot catch an error
shared by both torch-pme code paths (a wrong self/background correction, say); comparing
against a second implementation closes that gap.

The comparison is only meaningful if both sides run at the *same* Ewald parameters. Two
things are needed for that:

* the per-system smearing must be pinned. ``jaxpme.batched_tiled.Ewald.prepare`` applies
  one ``cutoff``/``smearing`` to the whole list, so this module goes through the
  per-structure ``jaxpme.batched_tiled.batching.prepare`` instead and feeds it the
  smearings the torch-pme tiling derived.
* the real-space cutoff must be pinned. torch-pme cuts off at
  :math:`5\sigma = 10\lambda`, jax-pme at :math:`8\lambda`; left to its default,
  jax-pme's real-space truncation error alone would dominate the comparison.

With that, 3D-periodic systems agree to machine precision. The 2D slabs agree to
:math:`\approx 3\cdot 10^{-6}`, which is a genuine difference between the two slab
(Yeh--Berkowitz) corrections rather than a parameter mismatch -- it survives handing both
codes an already-shrunk cell -- but it stays well inside the residual
:math:`\approx e^{-3\pi}` of the slab treatment itself, so neither result is wrong.

`jax-pme` is an optional dependency; this module is skipped when it is missing. It lives
at https://github.com/lab-cosmo/jax-pme and the tiled backend needs the branch of
https://github.com/lab-cosmo/jax-pme/pull/29::

    pip install jax
    pip install git+https://github.com/lab-cosmo/jax-pme.git@add/batched-tiled
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read

from torchpme import CoulombPotential, EwaldCalculator
from torchpme.lib import ewald_params_from_num_k, prepare_tiled_batch, shrink_2d_cell

sys.path.append(str(Path(__file__).parents[1]))
from helpers import MAD_CAT_FRAMES, MAD_SOL_FRAMES, mad_sample, neighbor_list

jax = pytest.importorskip("jax")
jaxpme_batching = pytest.importorskip("jaxpme.batched_tiled.batching")
jaxpme_tiled = pytest.importorskip("jaxpme.batched_tiled")

# jax computes in single precision unless told otherwise, which would put the comparison
# far above the tolerances below
jax.config.update("jax_enable_x64", True)

DTYPE = torch.float64
NUM_K = 200
MAD_FRAMES = 4  # how many frames of each MAD subset to run


def load_frames(frames_path, n_frames):
    """
    Load ``n_frames`` frames of a MAD subset for both implementations.

    Returns the ASE frames, carrying the charges :func:`helpers.mad_sample` synthesizes
    (the MAD frames store none), alongside the matching torch-pme samples.
    """
    frames = read(frames_path, f":{n_frames}")
    samples = [
        mad_sample(frames_path, index, NUM_K, dtype=DTYPE) for index in range(n_frames)
    ]
    for frame, sample in zip(frames, samples, strict=True):
        frame.set_initial_charges(sample["charges"][:, 0].numpy())
    return frames, samples


def sample_cutoff(sample):
    """
    The cutoff a sample's neighbor list was built at.

    :func:`helpers.mad_sample` sizes it on the *effective* cell, which for the 2D slabs
    of ``mad_cat`` is the vacuum-shrunk one, so the shrink has to be repeated here.
    """
    effective_cell = shrink_2d_cell(
        sample["cell"], sample["periodic"], sample["positions"]
    )
    _, _, cutoff = ewald_params_from_num_k(effective_cell, sample["periodic"], NUM_K)
    return float(cutoff)


def collate(samples):
    return prepare_tiled_batch(
        positions=[s["positions"] for s in samples],
        charges=[s["charges"] for s in samples],
        cells=[s["cell"] for s in samples],
        periodic=[s["periodic"] for s in samples],
        neighbor_indices=[s["neighbor_indices"] for s in samples],
        neighbor_distances=[s["neighbor_distances"] for s in samples],
        num_k=NUM_K,
    )


def jax_batch(frames, samples, tiling):
    """
    Collate the same systems for ``jaxpme.batched_tiled``.

    The smearings come from ``tiling`` and the cutoffs from the same helper that built
    the neighbor lists, so both implementations run at identical parameters.
    """
    prepared = []
    for index, (frame, sample) in enumerate(zip(frames, samples, strict=True)):
        prepared.append(
            jaxpme_batching.prepare(
                frame,
                num_k=NUM_K,
                cutoff=sample_cutoff(sample),
                smearing=float(tiling["sigma"][index]),
            )
        )
    return jaxpme_batching.get_batch(prepared)


def tiled_potential(batch, tiling, positions=None, distances=None):
    calculator = EwaldCalculator(
        # the constructor smearing / lr_wavelength are overridden per system by the tiling
        CoulombPotential(smearing=1.0),
        lr_wavelength=1.0,
    ).to(DTYPE)
    return calculator.forward_batched(
        batch["charges"],
        batch["cell"],
        batch["positions"] if positions is None else positions,
        batch["neighbor_indices"],
        batch["neighbor_distances"] if distances is None else distances,
        batch["system_index"],
        batch["periodic"],
        tiling,
    )


def tiled_energies(batch, potential, n_systems):
    energies = (batch["charges"] * potential).sum(dim=-1)
    return torch.zeros(n_systems, dtype=DTYPE).index_add_(
        0, batch["system_index"], energies
    )


def unpad(values, sr_batch):
    """
    Strip jax-pme's padding slots.

    Its flat atom array is the systems concatenated in input order followed by padding,
    so the masked array lines up element-for-element with the torch-pme batch.
    """
    mask = np.asarray(sr_batch.atom_mask).astype(bool)
    return torch.tensor(np.asarray(values)[mask], dtype=DTYPE)


def in_graph_distances(positions, samples):
    """
    Rebuild the pair distances from ``positions`` so that forces can flow.

    ``batch["neighbor_distances"]`` is a constant as far as autograd is concerned, so the
    tiled path needs the distances recomputed from the batched positions instead.
    """
    distances = []
    offset = 0
    for sample in samples:
        indices = sample["neighbor_indices"]
        _, shifts = neighbor_list(
            positions=sample["positions"],
            periodic=bool(sample["periodic"].any()),
            box=sample["cell"],
            cutoff=sample_cutoff(sample),
            neighbor_shifts=True,
        )
        vectors = (
            positions[offset + indices[:, 0]]
            - positions[offset + indices[:, 1]]
            - shifts @ sample["cell"]
        )
        distances.append(vectors.norm(dim=-1))
        offset += sample["positions"].shape[0]
    return torch.cat(distances)


def compare_energies(frames_paths, rtol, atol=0.0):
    frames, samples = [], []
    for frames_path, n_frames in frames_paths:
        frames_part, samples_part = load_frames(frames_path, n_frames)
        frames += frames_part
        samples += samples_part

    batch, tiling = collate(samples)
    potential = tiled_potential(batch, tiling)
    assert torch.all(torch.isfinite(potential))
    tiled = tiled_energies(batch, potential, len(samples))

    inputs = jax_batch(frames, samples, tiling)
    reference = torch.tensor(
        np.asarray(jaxpme_tiled.Ewald().energy(*inputs))[: len(samples)], dtype=DTYPE
    )
    torch.testing.assert_close(tiled, reference, rtol=rtol, atol=atol)


def compare_potentials(frames_path, n_frames, rtol, atol):
    frames, samples = load_frames(frames_path, n_frames)
    batch, tiling = collate(samples)
    tiled = tiled_potential(batch, tiling)

    inputs = jax_batch(frames, samples, tiling)
    reference = jaxpme_tiled.Ewald().potentials(*inputs)
    reference = unpad(reference, inputs[1]).reshape(-1, 1)
    torch.testing.assert_close(tiled, reference, rtol=rtol, atol=atol)


def compare_forces(frames_path, n_frames, rtol, atol):
    frames, samples = load_frames(frames_path, n_frames)
    batch, tiling = collate(samples)

    positions = batch["positions"].clone().requires_grad_(True)
    potential = tiled_potential(
        batch,
        tiling,
        positions=positions,
        distances=in_graph_distances(positions, samples),
    )
    energy = (batch["charges"] * potential).sum()
    (tiled,) = torch.autograd.grad(-energy, positions)

    inputs = jax_batch(frames, samples, tiling)
    _, reference = jaxpme_tiled.Ewald().energy_forces(*inputs)
    torch.testing.assert_close(tiled, unpad(reference, inputs[1]), rtol=rtol, atol=atol)


def test_jaxpme_energies_mad_sol():
    compare_energies([(MAD_SOL_FRAMES, MAD_FRAMES)], rtol=1e-10)


def test_jaxpme_potentials_mad_sol():
    compare_potentials(MAD_SOL_FRAMES, MAD_FRAMES, rtol=1e-10, atol=1e-12)


def test_jaxpme_forces_mad_sol():
    compare_forces(MAD_SOL_FRAMES, 2, rtol=1e-8, atol=1e-10)


# The 2D tolerances are far looser than the 3D ones because the two slab corrections
# differ. The difference is not a parameter mismatch -- it survives feeding both codes an
# already vacuum-shrunk cell -- but it stays well inside the ~exp(-3 pi) residual of the
# slab treatment itself, so neither result is wrong. It shows up per atom as a roughly
# constant offset (2e-7 on the potentials, 5e-7 on the forces), which is what the ``atol``
# floors below express; the ``rtol`` is what still catches a gross error.
def test_jaxpme_energies_mad_cat():
    compare_energies([(MAD_CAT_FRAMES, MAD_FRAMES)], rtol=1e-5)


def test_jaxpme_potentials_mad_cat():
    compare_potentials(MAD_CAT_FRAMES, MAD_FRAMES, rtol=1e-5, atol=2e-6)


def test_jaxpme_forces_mad_cat():
    compare_forces(MAD_CAT_FRAMES, 2, rtol=1e-4, atol=5e-6)


def test_jaxpme_mixed_batch():
    """3D cells and 2D slabs in one batch -- the case the tiling exists for."""
    compare_energies([(MAD_SOL_FRAMES, 2), (MAD_CAT_FRAMES, 2)], rtol=1e-5)
