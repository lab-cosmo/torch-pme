import pytest
import torch

from torchpme.tuning.ewald import EwaldErrorBounds
from torchpme.tuning.p3m import P3MErrorBounds
from torchpme.tuning.pme import PMEErrorBounds


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
    print(float(error_bound(**params)))
    torch.testing.assert_close(error_bound(**params), ref_err)


@pytest.mark.parametrize("charge_scale", [0.0, 0.25, 1.0, -3.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ewald_kspace_error_prefactor(charge_scale, dtype):
    # Eq. (S66) of the torch-pme paper and the documented Ewald estimate use
    # sum(q_i**2) / sqrt(N), not its square root, as the charge prefactor.
    charges = charge_scale * torch.tensor([[1.0], [-2.0], [1.0]], dtype=dtype)
    positions = torch.zeros((3, 3), dtype=dtype)
    cell = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=dtype))
    error_bound = EwaldErrorBounds(charges, cell, positions)
    smearing = torch.tensor(0.3, dtype=dtype)
    wavelength = torch.tensor(1.5, dtype=dtype)
    expected = (
        (6 * charge_scale**2 / 3**0.5)
        / (torch.pi * 0.3 * (24 / 1.5) ** 0.5)
        * torch.exp(torch.tensor(-2 * (torch.pi * 0.3 / 1.5) ** 2, dtype=dtype))
    )
    torch.testing.assert_close(error_bound.err_kspace(smearing, wavelength), expected)


@pytest.mark.parametrize("scale", [0.25, -3.0])
def test_ewald_error_scales_quadratically_with_charges(scale):
    charges = torch.tensor([[1.0], [-1.0]])
    positions = torch.zeros((2, 3))
    cell = 2 * torch.eye(3)
    original = EwaldErrorBounds(charges, cell, positions)
    scaled = EwaldErrorBounds(scale * charges, cell, positions)
    params = dict(smearing=0.3, lr_wavelength=1.5, cutoff=1.0)
    torch.testing.assert_close(scaled(**params), scale**2 * original(**params))
