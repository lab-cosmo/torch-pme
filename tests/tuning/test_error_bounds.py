import sys
from pathlib import Path

import pytest
import torch

from torchpme import CoulombPotential, EwaldCalculator
from torchpme.tuning.ewald import EwaldErrorBounds
from torchpme.tuning.p3m import P3MErrorBounds
from torchpme.tuning.pme import PMEErrorBounds

sys.path.append(str(Path(__file__).parents[1]))
from helpers import neighbor_list


@pytest.mark.parametrize(
    ("error_bound", "params", "ref_err"),
    [
        (
            EwaldErrorBounds,
            dict(smearing=1.0, lr_wavelength=0.5, cutoff=4.4),
            torch.tensor(8.4304e-05),
        ),
        (
            PMEErrorBounds,
            dict(smearing=1.0, mesh_spacing=0.5, cutoff=4.4, interpolation_nodes=3),
            torch.tensor(0.0011180),
        ),
        (
            P3MErrorBounds,
            dict(smearing=1.0, mesh_spacing=0.5, cutoff=4.4, interpolation_nodes=3),
            torch.tensor(4.5961e-04),
        ),
    ],
)
def test_error_bounds(error_bound, params, ref_err):
    charges = torch.tensor([[1.0], [-1.0]])
    cell = torch.eye(3)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]])
    error_bound = error_bound(charges, cell, positions)
    torch.testing.assert_close(error_bound(**params), ref_err)


@pytest.mark.parametrize(
    ("error_bound", "params"),
    [
        (
            EwaldErrorBounds,
            dict(smearing=0.5, lr_wavelength=0.5, cutoff=1.5),
        ),
        (
            PMEErrorBounds,
            dict(smearing=0.5, mesh_spacing=0.5, cutoff=1.5, interpolation_nodes=3),
        ),
        (
            P3MErrorBounds,
            dict(smearing=0.5, mesh_spacing=0.5, cutoff=1.5, interpolation_nodes=3),
        ),
    ],
)
@pytest.mark.parametrize(("charge_scale", "prefactor"), [(2.0, 1.0), (1.0, 14.4)])
def test_error_bounds_scale_with_prefactor_times_charge_squared(
    error_bound, params, charge_scale, prefactor
):
    """A force error must carry the units of ``prefactor * q**2``, as in ESPResSo."""
    charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 2.0
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.7, 0.9]], dtype=torch.float64)

    reference = error_bound(charges, cell, positions)(**params)
    scaled = error_bound(charge_scale * charges, cell, positions, prefactor)(**params)

    torch.testing.assert_close(scaled, prefactor * charge_scale**2 * reference)


def test_ewald_kspace_error_bounds_actual_force_error():
    """
    The Ewald k-space bound must be an upper bound of the actual RMS force error,
    but not a uselessly loose one.
    """
    torch.manual_seed(0)
    n_atoms, box_length, smearing, cutoff = 64, 6.0, 0.6, 2.95
    cell = torch.eye(3, dtype=torch.float64) * box_length
    positions = torch.rand(n_atoms, 3, dtype=torch.float64) * box_length
    charges = torch.ones(n_atoms, 1, dtype=torch.float64)
    charges[n_atoms // 2 :] = -1.0
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=positions, box=cell, cutoff=cutoff
    )

    def forces(lr_wavelength):
        pos = positions.clone().requires_grad_(True)
        calculator = EwaldCalculator(
            CoulombPotential(smearing=smearing), lr_wavelength=lr_wavelength
        )
        # the fixed `neighbor_distances` drop the real-space forces, which are the
        # same for every lr_wavelength and would cancel in the difference anyway
        potential = calculator(charges, cell, pos, neighbor_indices, neighbor_distances)
        return -torch.autograd.grad((charges * potential).sum(), pos)[0]

    reference = forces(0.5)
    error_bounds = EwaldErrorBounds(charges, cell, positions)
    for lr_wavelength in (1.6, 1.2):
        actual = (forces(lr_wavelength) - reference).pow(2).sum(dim=1).mean().sqrt()
        bound = error_bounds.err_kspace(
            torch.tensor(smearing), torch.tensor(lr_wavelength)
        )
        assert actual <= bound <= 30 * actual
