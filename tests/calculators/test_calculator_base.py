import pytest
import torch

from meshlode.calculators.base import CalculatorBaseTorch


# Define some example parameters
dtype = torch.float32
device = "cpu"
charges_1 = torch.ones((4, 1), dtype=dtype, device=device)
positions_1 = 0.3 * torch.arange(12, dtype=dtype, device=device).reshape((4, 3))
charges_2 = torch.ones((5, 3), dtype=dtype, device=device)
positions_2 = 0.7 * torch.arange(15, dtype=dtype, device=device).reshape((5, 3))
cell_1 = torch.eye(3, dtype=dtype, device=device)
cell_2 = torch.arange(9, dtype=dtype, device=device).reshape((3, 3))


class TestCalculator(CalculatorBaseTorch):
    def compute(self, positions, cell, charges, neighbor_indices, neighbor_shifts):
        return self._compute_impl(
            positions=positions,
            cell=cell,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

    def forward(self, positions, cell, charges, neighbor_indices, neighbor_shifts):
        return self._compute_impl(
            positions=positions,
            cell=cell,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

    def _compute_single_system(
        self, positions, cell, charges, neighbor_indices, neighbor_shifts
    ):
        return charges


@pytest.mark.parametrize("method_name", ["compute", "forward"])
@pytest.mark.parametrize(
    "positions, charges",
    [
        (torch.ones([2, 3]), torch.ones(2).reshape((-1, 1))),
        ([torch.ones([2, 3])], [torch.ones(2).reshape((-1, 1))]),
        (
            [torch.ones([2, 3]), torch.ones([4, 3])],
            [torch.ones(2).reshape((-1, 1)), torch.ones(4).reshape((-1, 1))],
        ),
    ],
)
def test_compute_output_shapes(method_name, positions, charges):
    calculator = TestCalculator(exponent=1.0)
    method = getattr(calculator, method_name)

    result = method(
        positions=positions,
        cell=None,
        charges=charges,
        neighbor_indices=None,
        neighbor_shifts=None,
    )
    if type(result) is list:
        for charge_single, result_single in zip(charges, result):
            assert result_single.shape == charge_single.shape
    else:
        if type(charges) is list:
            charges = charges[0]
        assert result.shape == charges.shape


# Tests for a mismatch in the number of provided inputs for different variables
def test_mismatched_numbers_cell():
    calculator = TestCalculator(exponent=1.0)
    match = r"Got inconsistent numbers of positions \(2\) and cell \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[positions_1, positions_2],
            cell=[cell_1, cell_2, torch.eye(3)],
            charges=[charges_1, charges_2],
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_mismatched_numbers_charges():
    calculator = TestCalculator(exponent=1.0)
    match = r"Got inconsistent numbers of positions \(2\) and charges \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[positions_1, positions_2],
            cell=None,
            charges=[charges_1, charges_2, charges_2],
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_mismatched_numbers_neighbor_indices():
    calculator = TestCalculator(exponent=1.0)
    match = r"Got inconsistent numbers of positions \(2\) and neighbor_indices \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[positions_1, positions_2],
            cell=None,
            charges=[charges_1, charges_2],
            neighbor_indices=[charges_1, charges_2, positions_1],
            neighbor_shifts=None,
        )


def test_mismatched_numbers_neighbor_shiftss():
    calculator = TestCalculator(exponent=1.0)
    match = r"Got inconsistent numbers of positions \(2\) and neighbor_shifts \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[positions_1, positions_2],
            cell=None,
            charges=[charges_1, charges_2],
            neighbor_indices=None,
            neighbor_shifts=[charges_1, charges_2, positions_1],
        )


# Tests for invalid shape, dtype and device of positions
def test_invalid_shape_positions():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `positions` must be a \(n_atoms x 3\) tensor, got at least "
        r"one tensor with shape \(4, 5\)"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=torch.ones((4, 5), dtype=dtype, device=device),
            cell=None,
            charges=charges_1,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_dtype_positions():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `positions` must have the same type torch.float32 as the "
        r"first provided one. Got at least one tensor of type "
        r"torch.float64"
    )
    positions_2_wrong_dtype = torch.ones((5, 3), dtype=torch.float64, device=device)
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[positions_1, positions_2_wrong_dtype],
            cell=None,
            charges=[charges_1, charges_2],
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_device_positions():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `positions` must be on the same device cpu as the "
        r"first provided one. Got at least one tensor on device "
        r"meta"
    )
    positions_2_wrong_device = torch.ones((5, 3), dtype=dtype, device="meta")
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[positions_1, positions_2_wrong_device],
            cell=None,
            charges=[charges_1, charges_2],
            neighbor_indices=None,
            neighbor_shifts=None,
        )


# Tests for invalid shape, dtype and device of cell
def test_invalid_shape_cell():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `cell` must be a \(3 x 3\) tensor, got at least one tensor with "
        r"shape \(2, 2\)"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=torch.ones([2, 2], dtype=dtype, device=device),
            charges=charges_1,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_dtype_cell():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `cell` must have the same type torch.float32 as positions, "
        r"got at least one tensor of type torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=torch.ones([3, 3], dtype=torch.float64, device=device),
            charges=charges_1,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_device_cell():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `cell` must be on the same device cpu as positions, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=torch.ones([3, 3], dtype=dtype, device="meta"),
            charges=charges_1,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


# Tests for invalid shape, dtype and device of charges
def test_invalid_dim_charges():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `charges` needs to be a 2-dimensional tensor, got at least "
        r"one tensor with 1 dimension\(s\) and shape "
        r"\(4,\)"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=torch.ones(len(positions_1), dtype=dtype, device=device),
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_shape_charges():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `charges` must be a \(n_atoms x n_channels\) tensor, with"
        r"`n_atoms` being the same as the variable `positions`. Got at "
        r"least one tensor with shape \(6, 2\) where "
        r"positions contains 4 atoms"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=torch.ones((6, 2), dtype=dtype, device=device),
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_dtype_charges():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `charges` must have the same type torch.float32 as positions, "
        r"got at least one tensor of type torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=torch.ones((4, 2), dtype=torch.float64, device=device),
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_dtype_charges():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `charges` must be on the same device cpu as positions, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=torch.ones((4, 2), dtype=dtype, device="meta"),
            neighbor_indices=None,
            neighbor_shifts=None,
        )


# Tests for invalid shape, dtype and device of neighbor_indices and neighbor_shifts
def test_need_both_neighbor_indices_and_shifts():
    calculator = TestCalculator(exponent=1.0)
    match = r"Need to provide both neighbor_indices and neighbor_shifts together."
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=charges_1,
            neighbor_indices=torch.ones((2, 10), dtype=dtype, device=device),
            neighbor_shifts=None,
        )


def test_invalid_shape_neighbor_indices():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"neighbor_indices is expected to have shape \(2, num_neighbors\)"
        r", but got \(4, 10\) for one structure"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=charges_1,
            neighbor_indices=torch.ones((4, 10), dtype=dtype, device=device),
            neighbor_shifts=torch.ones((10, 3), dtype=dtype, device=device),
        )


def test_invalid_shape_neighbor_shifts():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"neighbor_shifts is expected to have shape \(num_neighbors, 3\)"
        r", but got \(10, 2\) for one structure"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=charges_1,
            neighbor_indices=torch.ones((2, 10), dtype=dtype, device=device),
            neighbor_shifts=torch.ones((10, 2), dtype=dtype, device=device),
        )


def test_invalid_shape_neighbor_shifts():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"`neighbor_indices` and `neighbor_shifts` need to have shapes "
        r"\(2, num_neighbors\) and \(num_neighbors, 3\). For at least one"
        r"structure, got \(2, 10\) and "
        r"\(11, 3\), which is inconsistent"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=charges_1,
            neighbor_indices=torch.ones((2, 10), dtype=dtype, device=device),
            neighbor_shifts=torch.ones((11, 3), dtype=dtype, device=device),
        )


def test_invalid_device_neighbor_indices():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `neighbor_indices` must be on the same device cpu as positions, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=charges_1,
            neighbor_indices=torch.ones((2, 10), dtype=dtype, device="meta"),
            neighbor_shifts=torch.ones((10, 3), dtype=dtype, device=device),
        )


def test_invalid_device_neighbor_shifts():
    calculator = TestCalculator(exponent=1.0)
    match = (
        r"each `neighbor_shifts` must be on the same device cpu as positions, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=positions_1,
            cell=None,
            charges=charges_1,
            neighbor_indices=torch.ones((2, 10), dtype=dtype, device=device),
            neighbor_shifts=torch.ones((10, 3), dtype=dtype, device="meta"),
        )
