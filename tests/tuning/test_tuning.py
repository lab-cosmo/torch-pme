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
from helpers import DEVICES, DTYPES, define_crystal, neighbor_list

DEFAULT_CUTOFF = 4.4


def system(device=None, dtype=None):
    charges = torch.ones((4, 1), dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device)
    positions = 0.3 * torch.arange(12, dtype=dtype, device=device).reshape((4, 3))

    return charges, cell, positions


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_TunerBase_init(device, dtype):
    """
    Check that `TunerBase` initilizes correctly.

    We are using dummy `neighbor_indices` and `neighbor_distances` to verify types. Have
    to be sure that these dummy variables are initilized correctly.
    """
    charges, cell, positions = system(device, dtype)
    TunerBase(
        charges=charges,
        cell=cell,
        positions=positions,
        cutoff=DEFAULT_CUTOFF,
        calculator=1.0,
        exponent=1,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("calculator", "tune", "param_length"),
    [
        (EwaldCalculator, tune_ewald, 1),
        (PMECalculator, tune_pme, 2),
        (P3MCalculator, tune_p3m, 2),
    ],
)
@pytest.mark.parametrize("accuracy", [1e-1, 1e-3, 1e-5])
@pytest.mark.parametrize("full_neighbor_list", [True, False])
def test_parameter_choose(
    device, dtype, calculator, tune, param_length, accuracy, full_neighbor_list
):
    """
    Check that the Madelung constants obtained from the Ewald sum calculator matches
    the reference values and that all branches of the from_accuracy method are covered.
    """
    # Get input parameters and adjust to account for scaling
    pos, charges, cell, madelung_ref, num_units = define_crystal(
        dtype=dtype, device=device
    )

    # Compute neighbor list
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=pos,
        box=cell,
        cutoff=DEFAULT_CUTOFF,
        full_neighbor_list=full_neighbor_list,
    )

    smearing, params, _ = tune(
        charges,
        cell,
        pos,
        DEFAULT_CUTOFF,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        full_neighbor_list=full_neighbor_list,
        accuracy=accuracy,
    )

    assert len(params) == param_length

    # Compute potential and compare against target value using default hypers
    calc = calculator(
        potential=(CoulombPotential(smearing=smearing)),
        full_neighbor_list=full_neighbor_list,
        **params,
    )
    calc.to(device=device, dtype=dtype)
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("full_nl_list", [True, False])
def test_cutoff_filter(device, dtype, full_neighbor_list):
    """
    Check that `TunerBase` initilizes correctly.

    We are using dummy `neighbor_indices` and `neighbor_distances` to verify types. Have
    to be sure that these dummy variables are initilized correctly.
    """
    _, cell, positions = system(device, dtype)
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=positions,
        box=cell,
        cutoff=DEFAULT_CUTOFF * 10,
        full_neighbor_list=full_neighbor_list,
    )
    _, filtered_distances = TunerBase.filter_neighbors(
        DEFAULT_CUTOFF, neighbor_indices, neighbor_distances
    )
    assert filtered_distances.max() < DEFAULT_CUTOFF

    _, distance_from_calculation = neighbor_list(
        positions=positions,
        box=cell,
        cutoff=DEFAULT_CUTOFF,
        full_neighbor_list=full_neighbor_list,
    )
    assert torch.allclose(filtered_distances, distance_from_calculation)


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_accuracy_error(tune):
    pos, charges, cell, _, _ = define_crystal()

    match = "'foo' is not a float."
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=pos, box=cell, cutoff=DEFAULT_CUTOFF
    )
    with pytest.raises(ValueError, match=match):
        tune(
            charges=charges,
            cell=cell,
            positions=pos,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            accuracy="foo",
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_exponent_not_1_error(tune):
    pos, charges, cell, _, _ = define_crystal()
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=pos, box=cell, cutoff=DEFAULT_CUTOFF
    )

    match = "Only exponent = 1 is supported but got 2."
    with pytest.raises(NotImplementedError, match=match):
        tune(
            charges=charges,
            cell=cell,
            positions=pos,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            exponent=2,
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_shape_positions(tune):
    charges, cell, _ = system()
    match = (
        r"`positions` must be a tensor with shape \[n_atoms, 3\], got tensor with "
        r"shape \[4, 5\]"
    )
    with pytest.raises(ValueError, match=match):
        tune(
            charges=charges,
            cell=cell,
            positions=torch.ones((4, 5)),
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=None,
            neighbor_distances=None,
        )


# Tests for invalid shape, dtype and device of cell
@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_shape_cell(tune):
    charges, _, positions = system()
    match = (
        r"`cell` must be a tensor with shape \[3, 3\], got tensor with shape \[2, 2\]"
    )
    with pytest.raises(ValueError, match=match):
        tune(
            charges=charges,
            cell=torch.ones([2, 2]),
            positions=positions,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=None,
            neighbor_distances=None,
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_cell(tune):
    charges, _, positions = system()
    match = (
        "provided `cell` has a determinant of 0 and therefore is not valid for "
        "periodic calculation"
    )
    with pytest.raises(ValueError, match=match):
        tune(
            charges=charges,
            cell=torch.zeros(3, 3),
            positions=positions,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=None,
            neighbor_distances=None,
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_dtype_cell(tune):
    charges, _, positions = system()
    match = r"type of `cell` \(torch.float64\) must be same as that of the `positions` class \(torch.float32\)"
    with pytest.raises(TypeError, match=match):
        tune(
            charges=charges,
            cell=torch.eye(3, dtype=torch.float64),
            positions=positions,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=None,
            neighbor_distances=None,
        )
