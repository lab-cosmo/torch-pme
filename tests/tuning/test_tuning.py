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
from torchpme._utils import _get_device, _get_dtype
from torchpme.tuning import tune_ewald, tune_p3m, tune_pme
from torchpme.tuning.tuner import TunerBase

sys.path.append(str(Path(__file__).parents[1]))
from helpers import define_crystal, neighbor_list

DEFAULT_CUTOFF = 4.4
DEVICES = ["cpu", torch.device("cpu")] + torch.cuda.is_available() * ["cuda"]
DTYPES = [torch.float32, torch.float64]


def system(device=None, dtype=None):
    device = _get_device(device)
    dtype = _get_dtype(dtype)

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
        dtype=dtype,
        device=device,
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
def test_parameter_choose(device, dtype, calculator, tune, param_length, accuracy):
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
        positions=pos, box=cell, cutoff=DEFAULT_CUTOFF
    )

    smearing, params, _ = tune(
        charges,
        cell,
        pos,
        DEFAULT_CUTOFF,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        accuracy=accuracy,
        dtype=dtype,
        device=device,
    )

    assert len(params) == param_length

    # Compute potential and compare against target value using default hypers
    calc = calculator(
        potential=(CoulombPotential(smearing=smearing, dtype=dtype, device=device)),
        dtype=dtype,
        device=device,
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
    match = (
        r"type of `cell` \(torch.float64\) must be same as the class \(torch.float32\)"
    )
    with pytest.raises(TypeError, match=match):
        tune(
            charges=charges,
            cell=torch.eye(3, dtype=torch.float64),
            positions=positions,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=None,
            neighbor_distances=None,
        )
