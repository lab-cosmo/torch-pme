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
