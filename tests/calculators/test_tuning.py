import pytest
import torch
from test_values_ewald import define_crystal
from utils import neighbor_list_torch

from torchpme import EwaldPotential, tune_ewald


@pytest.mark.parametrize(
    ("accuracy", "rtol"),
    [
        ("medium", 1e-3),
        ("accurate", 1e-6),
        (1e-1, 1e-1),
    ],
)
def test_parameter_choose(accuracy, rtol):
    """
    Check that the Madelung constants obtained from the Ewald sum calculator matches
    the reference values and that all branches of the from_accuracy method are covered.
    """
    # Get input parameters and adjust to account for scaling
    pos, charges, cell, madelung_ref, num_units = define_crystal()

    ewald_params, sr_cutoff = tune_ewald(pos, charges, cell, accuracy=accuracy)

    assert len(ewald_params) == 2

    # Compute neighbor list
    neighbor_indices, neighbor_distances = neighbor_list_torch(
        positions=pos, periodic=True, box=cell, cutoff=sr_cutoff
    )

    # Compute potential and compare against target value using default hypers
    calc = EwaldPotential(**ewald_params)
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


def test_paramaters_fast():
    pos, charges, cell, _, _ = define_crystal()

    ewald_params, sr_cutoff = tune_ewald(pos, charges, cell, accuracy="fast")

    smearing = len(pos) ** (1 / 6) / 2**0.5 * 1.3

    assert ewald_params["atomic_smearing"] == smearing
    assert ewald_params["lr_wavelength"] == 2 * torch.pi * smearing / 2.2
    assert sr_cutoff == smearing * 2.2


def test_accuracy_error():
    pos, charges, cell, _, _ = define_crystal()

    match = (
        "'foo' is not a valid method or a float: Choose from 'fast',"
        "'medium' or 'accurate', or provide a float for the accuracy."
    )
    with pytest.raises(ValueError, match=match):
        tune_ewald(pos, charges, cell, accuracy="foo")


def test_multi_charge_channel_error():
    pos, charges, cell, _, _ = define_crystal()
    charges = torch.hstack([charges, charges])

    match = "Found 2 charge channels, but only one iss supported"
    with pytest.raises(NotImplementedError, match=match):
        tune_ewald(pos, charges, cell, accuracy=None)
