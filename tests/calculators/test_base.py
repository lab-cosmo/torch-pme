import pytest
import torch

from torchpme.calculators import Calculator, estimate_smearing, get_cscl_data
from torchpme.lib.potentials import CoulombPotential

# Define some example parameters
DTYPE = torch.float32
DEVICE = "cpu"
CHARGES_1 = torch.ones((4, 1), dtype=DTYPE, device=DEVICE)
POSITIONS_1 = 0.3 * torch.arange(12, dtype=DTYPE, device=DEVICE).reshape((4, 3))
CELL_1 = torch.eye(3, dtype=DTYPE, device=DEVICE)
NEIGHBOR_INDICES = torch.ones(3, 2, dtype=int)
NEIGHBOR_DISTANCES = torch.ones(3)


# non-range-separated Coulomb direct calculator
class CalculatorTest(Calculator):
    def __init__(self):
        super().__init__(
            potential=CoulombPotential(smearing=None, exclusion_radius=None)
        )


def test_cscl_data():
    data = get_cscl_data()

    assert len(data) == 7
    for el in data:
        assert type(el) is torch.Tensor


def test_compute_output_shapes():
    """Test that output type matches the input type"""
    calculator = CalculatorTest()

    positions = POSITIONS_1
    charges = CHARGES_1
    cell = CELL_1
    neighbor_indices = NEIGHBOR_INDICES
    neighbor_distances = NEIGHBOR_DISTANCES

    result = calculator.forward(
        charges=charges,
        cell=cell,
        positions=positions,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )
    assert type(result) is torch.Tensor
    assert result.shape == charges.shape


# Tests for invalid shape, dtype and device of positions
def test_invalid_shape_positions():
    calculator = CalculatorTest()
    match = (
        r"each `positions` must be a tensor with shape \[n_atoms, 3\], got at least "
        r"one tensor with shape \[4, 5\]"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=torch.ones((4, 5), dtype=DTYPE, device=DEVICE),
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


# Tests for invalid shape, dtype and device of cell
def test_invalid_shape_cell():
    calculator = CalculatorTest()
    match = (
        r"each `cell` must be a tensor with shape \[3, 3\], got at least one tensor "
        r"with shape \[2, 2\]"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones([2, 2], dtype=DTYPE, device=DEVICE),
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_dtype_cell():
    calculator = CalculatorTest()
    match = (
        r"each `cell` must have the same type torch.float32 as `positions`, "
        r"got at least one tensor of type torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones([3, 3], dtype=torch.float64, device=DEVICE),
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_device_cell():
    calculator = CalculatorTest()
    match = (
        r"each `cell` must be on the same device cpu as `positions`, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones([3, 3], dtype=DTYPE, device="meta"),
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


# Tests for invalid shape, dtype and device of charges
def test_invalid_dim_charges():
    calculator = CalculatorTest()
    match = (
        r"each `charges` needs to be a 2-dimensional tensor, got at least "
        r"one tensor with 1 dimension\(s\) and shape "
        r"\[4\]"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=torch.ones(len(POSITIONS_1), dtype=DTYPE, device=DEVICE),
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_DISTANCES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_shape_charges():
    calculator = CalculatorTest()
    match = (
        r"each `charges` must be a tensor with shape \[n_atoms, n_channels\], with "
        r"`n_atoms` being the same as the variable `positions`. Got at "
        r"least one tensor with shape \[6, 2\] where "
        r"positions contains 4 atoms"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=torch.ones((6, 2), dtype=DTYPE, device=DEVICE),
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_dtype_charges():
    calculator = CalculatorTest()
    match = (
        r"each `charges` must have the same type torch.float32 as `positions`, "
        r"got at least one tensor of type torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=torch.ones((4, 2), dtype=torch.float64, device=DEVICE),
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_device_charges():
    calculator = CalculatorTest()
    match = (
        r"each `charges` must be on the same device cpu as `positions`, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=torch.ones((4, 2), dtype=DTYPE, device="meta"),
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_shape_neighbor_indices():
    calculator = CalculatorTest()
    match = (
        r"neighbor_indices is expected to have shape \[num_neighbors, 2\]"
        r", but got \[4, 10\] for one structure"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=torch.ones((4, 10), dtype=DTYPE, device=DEVICE),
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_shape_neighbor_indices_neighbor_distances():
    calculator = CalculatorTest()
    match = (
        r"`neighbor_indices` and `neighbor_distances` need to have shapes "
        r"\[num_neighbors, 2\] and \[num_neighbors\]. For at least one "
        r"structure, got \[10, 2\] and "
        r"\[11, 3\], which is inconsistent"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=torch.ones((10, 2), dtype=DTYPE, device=DEVICE),
            neighbor_distances=torch.ones((11, 3), dtype=DTYPE, device=DEVICE),
        )


def test_invalid_device_neighbor_indices():
    calculator = CalculatorTest()
    match = (
        r"each `neighbor_indices` must be on the same device cpu as `positions`, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=torch.ones((10, 2), dtype=DTYPE, device="meta"),
            neighbor_distances=torch.ones((10), dtype=DTYPE, device=DEVICE),
        )


def test_invalid_device_neighbor_distances():
    calculator = CalculatorTest()
    match = (
        r"each `neighbor_distances` must be on the same device cpu as `positions`, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=torch.ones((10, 2), dtype=DTYPE, device=DEVICE),
            neighbor_distances=torch.ones((10), dtype=DTYPE, device="meta"),
        )


def test_no_cell():
    match = (
        "provided `cell` has a determinant of 0 and therefore is not valid for "
        "periodic calculation"
    )
    with pytest.raises(ValueError, match=match):
        estimate_smearing(cell=torch.zeros(3, 3))
