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
from torchpme.utils.tuning.grid_search import grid_search

sys.path.append(str(Path(__file__).parents[1]))
from helpers import define_crystal, neighbor_list

DTYPE = torch.float32
DEVICE = "cpu"
DEFAULT_CUTOFF = 4.4
CHARGES_1 = torch.ones((4, 1), dtype=DTYPE, device=DEVICE)
POSITIONS_1 = 0.3 * torch.arange(12, dtype=DTYPE, device=DEVICE).reshape((4, 3))
CELL_1 = torch.eye(3, dtype=DTYPE, device=DEVICE)


@pytest.mark.parametrize(
    ("calculator", "method", "param_length"),
    [
        (EwaldCalculator, "ewald", 1),
        (PMECalculator, "pme", 2),
        (P3MCalculator, "p3m", 2),
    ],
)
@pytest.mark.parametrize("accuracy", [1e-1, 1e-3, 1e-5])
def test_parameter_choose(calculator, method, param_length, accuracy):
    """
    Check that the Madelung constants obtained from the Ewald sum calculator matches
    the reference values and that all branches of the from_accuracy method are covered.
    """
    # Get input parameters and adjust to account for scaling
    pos, charges, cell, madelung_ref, num_units = define_crystal()

    smearing, params, sr_cutoff = grid_search(
        method=method,
        charges=charges,
        cell=cell,
        positions=pos,
        cutoff=DEFAULT_CUTOFF,
        accuracy=accuracy,
    )

    assert len(params) == param_length

    # Compute neighbor list
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=pos, periodic=True, box=cell, cutoff=sr_cutoff
    )

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


"""def test_odd_interpolation_nodes():
    pos, charges, cell, madelung_ref, num_units = define_crystal()

    smearing, params, sr_cutoff = tune_pme(
        sum_squared_charges=float(torch.sum(charges**2)),
        cell=cell,
        positions=pos,
        interpolation_nodes=5,
        learning_rate=0.75,
    )

    neighbor_indices, neighbor_distances = neighbor_list(
        positions=pos, periodic=True, box=cell, cutoff=sr_cutoff
    )

    calc = PMECalculator(potential=CoulombPotential(smearing=smearing), **params)
    potentials = calc.forward(
        positions=pos,
        charges=charges,
        cell=cell,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )
    energies = potentials * charges
    madelung = -torch.sum(energies) / num_units

    torch.testing.assert_close(madelung, madelung_ref, atol=0, rtol=1e-3)"""


'''@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_fix_parameters(tune):
    """Test that the parameters are fixed when they are passed as arguments."""
    pos, charges, cell, _, _ = define_crystal()

    kwargs_ref = {
        "sum_squared_charges": float(torch.sum(charges**2)),
        "cell": cell,
        "positions": pos,
        "max_steps": 5,
    }

    kwargs = kwargs_ref.copy()
    kwargs["smearing"] = 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smearing, _, _ = tune(**kwargs)
    pytest.approx(smearing, 0.1)

    kwargs = kwargs_ref.copy()
    if tune.__name__ in ["tune_pme", "tune_p3m"]:
        kwargs["mesh_spacing"] = 0.1
    else:
        kwargs["lr_wavelength"] = 0.1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, kspace_param, _ = tune(**kwargs)

    kspace_param = list(kspace_param.values())[0]
    pytest.approx(kspace_param, 0.1)

    kwargs = kwargs_ref.copy()
    kwargs["cutoff"] = 0.1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, sr_cutoff = tune(**kwargs)
    pytest.approx(sr_cutoff, 1.0)'''


def test_accuracy_error():
    pos, charges, cell, _, _ = define_crystal()

    match = "'foo' is not a float."
    with pytest.raises(ValueError, match=match):
        grid_search("ewald", charges, cell, pos, DEFAULT_CUTOFF, accuracy="foo")


def test_exponent_not_1_error():
    pos, charges, cell, _, _ = define_crystal()

    match = "Only exponent = 1 is supported"
    with pytest.raises(NotImplementedError, match=match):
        grid_search("ewald", charges, cell, pos, DEFAULT_CUTOFF, exponent=2)


def test_invalid_shape_positions():
    match = (
        r"each `positions` must be a tensor with shape \[n_atoms, 3\], got at least "
        r"one tensor with shape \[4, 5\]"
    )
    with pytest.raises(ValueError, match=match):
        grid_search(
            "ewald",
            CHARGES_1,
            CELL_1,
            torch.ones((4, 5), dtype=DTYPE, device=DEVICE),
            DEFAULT_CUTOFF,
        )


# Tests for invalid shape, dtype and device of cell
def test_invalid_shape_cell():
    match = (
        r"each `cell` must be a tensor with shape \[3, 3\], got at least one tensor "
        r"with shape \[2, 2\]"
    )
    with pytest.raises(ValueError, match=match):
        grid_search(
            "ewald",
            CHARGES_1,
            torch.ones([2, 2], dtype=DTYPE, device=DEVICE),
            POSITIONS_1,
            DEFAULT_CUTOFF,
        )


def test_invalid_cell():
    match = (
        "provided `cell` has a determinant of 0 and therefore is not valid for "
        "periodic calculation"
    )
    with pytest.raises(ValueError, match=match):
        grid_search("ewald", CHARGES_1, torch.zeros(3, 3), POSITIONS_1, DEFAULT_CUTOFF)


def test_invalid_dtype_cell():
    match = (
        r"each `cell` must have the same type torch.float32 as `positions`, "
        r"got at least one tensor of type torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        grid_search(
            "ewald",
            CHARGES_1,
            torch.eye(3, dtype=torch.float64, device=DEVICE),
            POSITIONS_1,
            DEFAULT_CUTOFF,
        )


def test_invalid_device_cell():
    match = (
        r"each `cell` must be on the same device cpu as `positions`, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        grid_search(
            "ewald",
            CHARGES_1,
            torch.eye(3, dtype=DTYPE, device="meta"),
            POSITIONS_1,
            DEFAULT_CUTOFF,
        )
