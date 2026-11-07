import pytest
import torch
from torch.testing import assert_close

from torchpme import (
    CombinedPotential,
    CoulombPotential,
    EwaldCalculator,
    InversePowerLawPotential,
)


@pytest.mark.parametrize(
    "periodic",
    [
        None,
        [True, True, True],
        [False, True, True],
        [True, False, True],
        [True, True, False],
    ],
)
@pytest.mark.parametrize("neutral", [False, True])
def test_combined_ewald_periodic_correction(periodic, neutral):
    dtype = torch.float64
    positions = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=dtype, requires_grad=True
    )
    cell = torch.diag(torch.tensor([8.0, 9.0, 16.0], dtype=dtype))
    cell.requires_grad_()
    second_charge = [-1.0, -2.0] if neutral else [0.5, -1.0]
    charges = torch.tensor([[1.0, 2.0], second_charge], dtype=dtype, requires_grad=True)
    inputs = {
        "positions": positions,
        "cell": cell,
        "charges": charges,
        "neighbor_indices": torch.tensor([[0, 1]]),
        "neighbor_distances": torch.linalg.norm(positions[1:] - positions[:1], dim=1),
        "periodic": None if periodic is None else torch.tensor(periodic),
    }
    potentials = [
        CoulombPotential(smearing=0.8, prefactor=2.0),
        InversePowerLawPotential(exponent=1, smearing=1.3, prefactor=0.75),
        InversePowerLawPotential(exponent=2, smearing=1.0),
    ]
    combined = CombinedPotential(
        potentials,
        initial_weights=torch.tensor([1.5, -0.25, 0.7], dtype=dtype),
        smearing=1.0,
    ).to(dtype=dtype)
    calculator = EwaldCalculator(combined, lr_wavelength=1.0)
    actual = calculator(**inputs)
    expected = sum(
        weight * EwaldCalculator(potential, lr_wavelength=1.0)(**inputs)
        for weight, potential in zip(combined.weights, potentials, strict=True)
    )
    assert_close(actual, expected)

    variables = (positions, cell, charges, combined.weights)
    actual_gradients = torch.autograd.grad(
        (charges * actual).sum(), variables, retain_graph=True
    )
    expected_gradients = torch.autograd.grad((charges * expected).sum(), variables)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert_close(actual_gradient, expected_gradient)
