import pytest
import torch
from test_values_ewald import define_crystal
from utils import neighbor_list_torch

from torchpme import EwaldPotential


@pytest.mark.parametrize(
    "optimize, accuracy, rtol, expected_smearing,"
    "expected_lr_wavelength, expected_cutoff",
    [
        (
            "fast",
            None,
            1,
            0.7071,
            4.4429,
            0.7071,
        ),  # fast branch, no requirement for the energy accuracy
        ("medium", None, 1e-3, 0.1676, 0.0732, 0.6907),  # medium mode
        ("accurate", None, 1e-6, 0.1948, 0.0525, 1.0731),  # accurate mode
        (None, 1e-3, 1e-3, 0.1676, 0.0732, 0.6907),  # accuracy provided branch
    ],
)
def test_parameter_choose(
    optimize, accuracy, rtol, expected_smearing, expected_lr_wavelength, expected_cutoff
):
    """
    Check that the Madelung constants obtained from the Ewald sum calculator matches
    the reference values and that all branches of the from_accuracy method are covered.
    """
    # Get input parameters and adjust to account for scaling
    pos, charges, cell, madelung_ref, num_units = define_crystal()
    charges = charges.reshape((-1, 1))

    # Define calculator and tolerances
    calc, sr_cutoff = EwaldPotential.from_accuracy(
        optimize, accuracy, pos, charges, cell
    )

    # Check smearing, lr_wavelength sr_cutoff values
    assert torch.isclose(
        calc.atomic_smearing, torch.tensor(expected_smearing), rtol=1e-3
    )
    assert torch.isclose(
        calc.lr_wavelength, torch.tensor(expected_lr_wavelength), rtol=1e-3
    )
    assert torch.isclose(sr_cutoff, torch.tensor(expected_cutoff), rtol=1e-3)

    # Compute neighbor list
    neighbor_indices, neighbor_distances = neighbor_list_torch(
        positions=pos, periodic=True, box=cell, cutoff=sr_cutoff
    )

    # Compute potential and compare against target value using default hypers
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


@pytest.mark.parametrize(
    "optimize, accuracy, error",
    [
        ("fine", None, "`optimize` must be one of 'fast', 'medium' or 'accurate'"),
        # `accuracy` is not given but `optimize` is not a valid value (ValueError case)
        (None, None, "Either `optimize` or `accuracy` must be set"),
        # neither optimize nor accuracy is provided (ValueError case)
    ],
)
def test_value_error(optimize, accuracy, error):
    """
    Test that ValueError is raised when neither optimize nor accuracy is provided.
    """
    pos, charges, cell, madelung_ref, num_units = define_crystal()
    charges = charges.reshape((-1, 1))

    with pytest.raises(ValueError, match=error):
        EwaldPotential.from_accuracy(optimize, accuracy, pos, charges, cell)


@pytest.mark.parametrize(
    "optimize, accuracy",
    [
        ("medium", 1e-3),  # Trigger warning
        ("accurate", 1e-6),  # Trigger warning
    ],
)
def test_warning_optimization_ignored(optimize, accuracy):
    """
    Test that a warning is raised when both `optimize` and `accuracy` are provided.
    """
    pos, charges, cell, madelung_ref, num_units = define_crystal()
    charges = charges.reshape((-1, 1))

    with pytest.warns(UserWarning, match="`optimize` is ignored if `accuracy` is set"):
        EwaldPotential.from_accuracy(optimize, accuracy, pos, charges, cell)
