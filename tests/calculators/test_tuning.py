import pytest
import torch
from test_values_ewald import define_crystal
from utils import neighbor_list_torch

from torchpme import (
    CoulombPotential,
    EwaldCalculator,
    PMECalculator,
)
from torchpme.utils.tuning import tune_ewald, tune_pme


@pytest.mark.parametrize(
    ("calculator", "tune", "param_length"),
    [
        (EwaldCalculator, tune_ewald, 1),
        (PMECalculator, tune_pme, 2),
    ],
)
@pytest.mark.parametrize(
    ("accuracy", "rtol"),
    [
        (1e-3, 1e-3),
        (1e-3, 2e-6),
        (1e-1, 1e-1),
    ],
)
def test_parameter_choose(calculator, tune, param_length, accuracy, rtol):
    """
    Check that the Madelung constants obtained from the Ewald sum calculator matches
    the reference values and that all branches of the from_accuracy method are covered.
    """
    # Get input parameters and adjust to account for scaling
    pos, charges, cell, madelung_ref, num_units = define_crystal()

    smearing, params, sr_cutoff = tune(
        torch.sum(charges**2, dim=0), cell, pos, accuracy=accuracy
    )

    assert len(params) == param_length

    # Compute neighbor list
    neighbor_indices, neighbor_distances = neighbor_list_torch(
        positions=pos, periodic=True, box=cell, cutoff=sr_cutoff
    )

    # Compute potential and compare against target value using default hypers
    calc = calculator(
        potential=CoulombPotential(smearing=smearing),
        **params,
    )
    potentials = calc.forward(
        positions=pos,
        charges=charges,
        cell=cell,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )
    energies = potentials * charges
    madelung = -torch.sum(energies) / num_units

    torch.testing.assert_close(madelung, madelung_ref, atol=0, rtol=rtol)


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme])
def test_accuracy_error(tune):
    pos, charges, cell, _, _ = define_crystal()

    match = (
        "'foo' is not a float."
    )
    with pytest.raises(ValueError, match=match):
        tune(torch.sum(charges**2, dim=0), cell, pos, accuracy="foo")


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme])
def test_multi_charge_channel_error(tune):
    pos, charges, cell, _, _ = define_crystal()
    charges = torch.hstack([charges, charges])

    match = "Found 2 charge channels, but only one is supported"
    with pytest.raises(NotImplementedError, match=match):
        tune(torch.sum(charges**2, dim=0), cell, pos, accuracy=None)
