import sys
from pathlib import Path

import pytest
import torch

from torchpme import (
    CoulombPotential,
    EwaldCalculator,
    P3MCalculator,
    PMECalculator,
)
from torchpme.tuning import tune_ewald, tune_p3m, tune_pme
from torchpme.tuning.tuner import TunerBase

sys.path.append(str(Path(__file__).parents[1]))
from helpers import compute_distances, define_crystal, neighbor_list

DTYPE = torch.float32
DEVICE = "cpu"
DEFAULT_CUTOFF = 4.4
CHARGES_1 = torch.ones((4, 1), dtype=DTYPE, device=DEVICE)
POSITIONS_1 = 0.3 * torch.arange(12, dtype=DTYPE, device=DEVICE).reshape((4, 3))
CELL_1 = torch.eye(3, dtype=DTYPE, device=DEVICE)


def _nl_calculation(pos, cell):
    neighbor_indices, neighbor_shifts = neighbor_list(
        positions=pos,
        periodic=True,
        box=cell,
        cutoff=DEFAULT_CUTOFF,
        neighbor_shifts=True,
    )

    neighbor_distances = compute_distances(
        positions=pos,
        neighbor_indices=neighbor_indices,
        cell=cell,
        neighbor_shifts=neighbor_shifts,
    )

    return neighbor_indices, neighbor_distances


def test_TunerBase_double():
    """
    Check that `TunerBase` initilizes with double precisions tensors.

    We are using dummy `neighbor_indices` and `neighbor_distances` to verify types. Have
    to be sure that these dummy variables are initilized correctly.
    """
    TunerBase(
        charges=CHARGES_1.to(dtype=torch.float64),
        cell=CELL_1.to(dtype=torch.float64),
        positions=POSITIONS_1.to(dtype=torch.float64),
        cutoff=DEFAULT_CUTOFF,
        calculator=1.0,
        exponent=1,
    )


@pytest.mark.parametrize(
    ("calculator", "tune", "param_length"),
    [
        (EwaldCalculator, tune_ewald, 1),
        (PMECalculator, tune_pme, 2),
        (P3MCalculator, tune_p3m, 2),
    ],
)
@pytest.mark.parametrize("accuracy", [1e-1, 1e-3, 1e-5])
def test_parameter_choose(calculator, tune, param_length, accuracy):
    """
    Check that the Madelung constants obtained from the Ewald sum calculator matches
    the reference values and that all branches of the from_accuracy method are covered.
    """
    # Get input parameters and adjust to account for scaling
    pos, charges, cell, madelung_ref, num_units = define_crystal()

    # Compute neighbor list
    neighbor_indices, neighbor_distances = _nl_calculation(pos, cell)

    smearing, params, _ = tune(
        charges,
        cell,
        pos,
        DEFAULT_CUTOFF,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        accuracy=accuracy,
    )

    assert len(params) == param_length

    # Compute potential and compare against target value using default hypers
    calc = calculator(
        potential=(CoulombPotential(smearing=smearing)),
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

    torch.testing.assert_close(madelung, madelung_ref, atol=0, rtol=accuracy)


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_accuracy_error(tune):
    pos, charges, cell, _, _ = define_crystal()

    match = "'foo' is not a float."
    neighbor_indices, neighbor_distances = _nl_calculation(pos, cell)
    with pytest.raises(ValueError, match=match):
        tune(
            charges,
            cell,
            pos,
            DEFAULT_CUTOFF,
            neighbor_indices,
            neighbor_distances,
            accuracy="foo",
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_exponent_not_1_error(tune):
    pos, charges, cell, _, _ = define_crystal()

    match = "Only exponent = 1 is supported but got 2."
    neighbor_indices, neighbor_distances = _nl_calculation(pos, cell)
    with pytest.raises(NotImplementedError, match=match):
        tune(
            charges,
            cell,
            pos,
            DEFAULT_CUTOFF,
            neighbor_indices,
            neighbor_distances,
            exponent=2,
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_shape_positions(tune):
    match = (
        r"`positions` must be a tensor with shape \[n_atoms, 3\], got tensor with "
        r"shape \[4, 5\]"
    )
    with pytest.raises(ValueError, match=match):
        tune(
            CHARGES_1,
            CELL_1,
            torch.ones((4, 5), dtype=DTYPE, device=DEVICE),
            DEFAULT_CUTOFF,
            None,  # dummy neighbor indices
            None,  # dummy neighbor distances
        )


# Tests for invalid shape, dtype and device of cell
@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_shape_cell(tune):
    match = (
        r"`cell` must be a tensor with shape \[3, 3\], got tensor with shape \[2, 2\]"
    )
    with pytest.raises(ValueError, match=match):
        tune(
            CHARGES_1,
            torch.ones([2, 2], dtype=DTYPE, device=DEVICE),
            POSITIONS_1,
            DEFAULT_CUTOFF,
            None,  # dummy neighbor indices
            None,  # dummy neighbor distances
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_cell(tune):
    match = (
        "provided `cell` has a determinant of 0 and therefore is not valid for "
        "periodic calculation"
    )
    with pytest.raises(ValueError, match=match):
        tune(CHARGES_1, torch.zeros(3, 3), POSITIONS_1, DEFAULT_CUTOFF, None, None)


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_dtype_cell(tune):
    match = (
        r"type of `cell` \(torch.float64\) must be same as `positions` "
        r"\(torch.float32\)"
    )
    with pytest.raises(ValueError, match=match):
        tune(
            CHARGES_1,
            torch.eye(3, dtype=torch.float64, device=DEVICE),
            POSITIONS_1,
            DEFAULT_CUTOFF,
            None,
            None,
        )
