import pytest
import torch

from torchpme.calculators import Calculator
from torchpme.lib.potentials import InversePowerLawPotential

# Define some example parameters
DTYPE = torch.float32
DEVICE = "cpu"
CHARGES_1 = torch.ones((4, 1), dtype=DTYPE, device=DEVICE)
POSITIONS_1 = 0.3 * torch.arange(12, dtype=DTYPE, device=DEVICE).reshape((4, 3))
CHARGES_2 = torch.ones((5, 3), dtype=DTYPE, device=DEVICE)
POSITIONS_2 = 0.7 * torch.arange(15, dtype=DTYPE, device=DEVICE).reshape((5, 3))
CELL_1 = torch.eye(3, dtype=DTYPE, device=DEVICE)
CELL_2 = torch.arange(9, dtype=DTYPE, device=DEVICE).reshape((3, 3))
NEIGHBOR_INDICES = torch.ones(3, 2)
NEIGHBOR_DISTANCES = torch.ones(3)


class PotentialTest(InversePowerLawPotential):
    def __init__(self):
        super().__init__(exponent=1.0, range_radius=1.0, cutoff_radius=1.0)


class CalculatorTest(Calculator):
    def __init__(self):
        super().__init__(potential=PotentialTest())

    def forward(self, charges, cell, positions, neighbor_indices, neighbor_distances):
        return charges


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


def test_type_check_error():
    calculator = CalculatorTest()

    for positions_type in [torch.Tensor, list]:
        for key in ["charges", "cell", "neighbor_indices", "neighbor_distances"]:
            kwargs = {
                "positions": positions_type([1]),
                "charges": positions_type([1]),
                "cell": positions_type([1]),
                "neighbor_indices": positions_type([1]),
                "neighbor_distances": positions_type([1]),
            }

            # Set key of interest to a different type then `positions`
            if positions_type is torch.Tensor:
                type_name = "torch.Tensor"
                item_type = "list"
                kwargs[key] = [1]
            else:
                type_name = "list"
                item_type = "torch.Tensor"
                kwargs[key] = torch.tensor([1])

            match = (
                f"Inconsistent parameter types. `positions` is a {type_name}, while "
                f"`{key}` is a {item_type}. Both need either be a list or a "
                "torch.Tensor!"
            )
            with pytest.raises(TypeError, match=match):
                calculator._validate_compute_parameters(**kwargs)


def test_exponent_out_of_range():
    match = r"`exponent` p=.* has to satisfy 0 < p <= 3"
    with pytest.raises(ValueError, match=match):
        CalculatorTest(exponent=-1)

    with pytest.raises(ValueError, match=match):
        CalculatorTest(exponent=4)


# Tests for a mismatch in the number of provided inputs for different variables
def test_mismatched_numbers_cell():
    calculator = CalculatorTest()
    match = r"Got inconsistent numbers of positions \(2\) and cell \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=[POSITIONS_1, POSITIONS_2],
            charges=[CHARGES_1, CHARGES_2],
            cell=[CELL_1, CELL_2, torch.eye(3)],
            neighbor_indices=[NEIGHBOR_INDICES, NEIGHBOR_INDICES],
            neighbor_distances=[NEIGHBOR_DISTANCES, NEIGHBOR_DISTANCES],
        )


def test_mismatched_numbers_charges():
    calculator = CalculatorTest()
    match = r"Got inconsistent numbers of positions \(2\) and charges \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=[POSITIONS_1, POSITIONS_2],
            charges=[CHARGES_1, CHARGES_2, CHARGES_2],
            cell=[CELL_1, CELL_2],
            neighbor_indices=[NEIGHBOR_INDICES, NEIGHBOR_INDICES],
            neighbor_distances=[NEIGHBOR_DISTANCES, NEIGHBOR_DISTANCES],
        )


def test_mismatched_numbers_neighbor_indices():
    calculator = CalculatorTest()
    match = r"Got inconsistent numbers of positions \(2\) and neighbor_indices \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=[POSITIONS_1, POSITIONS_2],
            charges=[CHARGES_1, CHARGES_2],
            cell=[CELL_1, CELL_2],
            neighbor_indices=[NEIGHBOR_INDICES, NEIGHBOR_INDICES, NEIGHBOR_INDICES],
            neighbor_distances=[NEIGHBOR_DISTANCES, NEIGHBOR_DISTANCES],
        )


def test_mismatched_numbers_neighbor_distances():
    calculator = CalculatorTest()
    match = r"Got inconsistent numbers of positions \(2\) and neighbor_distances \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=[POSITIONS_1, POSITIONS_2],
            charges=[CHARGES_1, CHARGES_2],
            cell=[CELL_1, CELL_2],
            neighbor_indices=[NEIGHBOR_INDICES, NEIGHBOR_INDICES],
            neighbor_distances=[
                NEIGHBOR_DISTANCES,
                NEIGHBOR_DISTANCES,
                NEIGHBOR_DISTANCES,
            ],
        )


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


def test_invalid_dtype_positions():
    calculator = CalculatorTest()
    match = (
        r"each `positions` must have the same type torch.float32 as the "
        r"first provided one. Got at least one tensor of type "
        r"torch.float64"
    )
    positions_2_wrong_dtype = torch.ones((5, 3), dtype=torch.float64, device=DEVICE)
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=[POSITIONS_1, positions_2_wrong_dtype],
            charges=[CHARGES_1, CHARGES_2],
            cell=[CELL_1, CELL_2],
            neighbor_indices=[NEIGHBOR_INDICES, NEIGHBOR_INDICES],
            neighbor_distances=[NEIGHBOR_DISTANCES, NEIGHBOR_DISTANCES],
        )


def test_invalid_device_positions():
    calculator = CalculatorTest()
    match = (
        r"each `positions` must be on the same device cpu as the "
        r"first provided one. Got at least one tensor on device "
        r"meta"
    )
    positions_2_wrong_device = POSITIONS_1.to(dtype=DTYPE, device="meta")
    with pytest.raises(ValueError, match=match):
        calculator.forward(
            positions=[POSITIONS_1, positions_2_wrong_device],
            charges=[CHARGES_1, CHARGES_2],
            cell=[CELL_1, CELL_2],
            neighbor_indices=[NEIGHBOR_INDICES, NEIGHBOR_INDICES],
            neighbor_distances=[NEIGHBOR_DISTANCES, NEIGHBOR_DISTANCES],
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
        Calculator.estimate_smearing(cell=torch.zeros(3, 3))
