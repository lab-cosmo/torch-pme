import pytest
import torch

from torchpme import CoulombPotential
from torchpme.calculators import Calculator

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


def test_wrong_device_positions():
    calculator = CalculatorTest()
    match = r"device of `positions` \(meta\) must be same as the class device \(cpu\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1.to(device="meta"),
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_wrong_dtype_positions():
    calculator = CalculatorTest()
    match = (
        r"type of `positions` \(torch.float64\) must be same as the class type "
        r"\(torch.float32\)"
    )
    with pytest.raises(TypeError, match=match):
        calculator.forward(
            positions=POSITIONS_1.to(dtype=torch.float64),
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


# Tests for invalid shape, dtype and device of positions
def test_invalid_shape_positions():
    calculator = CalculatorTest()
    match = (
        r"`positions` must be a tensor with shape \[n_atoms, 3\], got tensor with "
        r"shape \[4, 5\]"
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
        r"`cell` must be a tensor with shape \[3, 3\], got tensor with shape \[2, 2\]"
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
        r"type of `cell` \(torch.float64\) must be same as the class \(torch.float32\)"
    )
    with pytest.raises(TypeError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=CELL_1.to(dtype=torch.float64),
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_device_cell():
    calculator = CalculatorTest()
    match = r"device of `cell` \(meta\) must be same as the class \(cpu\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=CELL_1.to(device="meta"),
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_zero_cell():
    calculator = Calculator(
        potential=CoulombPotential(smearing=1, exclusion_radius=None)
    )
    match = (
        r"provided `cell` has a determinant of 0 and therefore is not valid for "
        r"periodic calculation"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.zeros([3, 3], dtype=DTYPE, device=DEVICE),
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


# Tests for invalid shape, dtype and device of charges
def test_invalid_dim_charges():
    calculator = CalculatorTest()
    match = (
        r"`charges` must be a 2-dimensional tensor, got tensor with 1 dimension\(s\) "
        r"and shape \[4\]"
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
        r"`charges` must be a tensor with shape \[n_atoms, n_channels\], with "
        r"`n_atoms` being the same as the variable `positions`. Got tensor with "
        r"shape \[6, 2\] where positions contains 4 atoms"
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
        r"type of `charges` \(torch.float64\) must be same as the class "
        r"\(torch.float32\)"
    )
    with pytest.raises(TypeError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1.to(dtype=torch.float64),
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_device_charges():
    calculator = CalculatorTest()
    match = r"device of `charges` \(meta\) must be same as the class \(cpu\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1.to(device="meta"),
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
        r"\[num_neighbors, 2\] and \[num_neighbors\], but got \[10, 2\] and \[11, 3\]"
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
    match = r"device of `neighbor_indices` \(meta\) must be same as the class \(cpu\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES.to(device="meta"),
            neighbor_distances=NEIGHBOR_DISTANCES,
        )


def test_invalid_device_neighbor_distances():
    calculator = CalculatorTest()
    match = r"device of `neighbor_distances` \(meta\) must be same as the class \(cpu\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES.to(device="meta"),
        )


def test_invalid_dtype_neighbor_distances():
    calculator = CalculatorTest()
    match = (
        r"type of `neighbor_distances` \(torch.float64\) must be same "
        r"as the class \(torch.float32\)"
    )
    with pytest.raises(TypeError, match=match):
        calculator.forward(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=CELL_1,
            neighbor_indices=NEIGHBOR_INDICES,
            neighbor_distances=NEIGHBOR_DISTANCES.to(dtype=torch.float64),
        )
