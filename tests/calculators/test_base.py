import pytest
import torch

from torchpme.calculators.base import CalculatorBaseTorch, PeriodicBase


# Define some example parameters
DTYPE = torch.float32
DEVICE = "cpu"
CHARGES_1 = torch.ones((4, 1), dtype=DTYPE, device=DEVICE)
POSITIONS_1 = 0.3 * torch.arange(12, dtype=DTYPE, device=DEVICE).reshape((4, 3))
CHARGES_2 = torch.ones((5, 3), dtype=DTYPE, device=DEVICE)
POSITIONS_2 = 0.7 * torch.arange(15, dtype=DTYPE, device=DEVICE).reshape((5, 3))
CELL_1 = torch.eye(3, dtype=DTYPE, device=DEVICE)
CELL_2 = torch.arange(9, dtype=DTYPE, device=DEVICE).reshape((3, 3))


class CalculatorTest(CalculatorBaseTorch):
    def compute(self, positions, charges, cell, neighbor_indices, neighbor_shifts):
        return self._compute_impl(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

    def forward(self, positions, charges, cell, neighbor_indices, neighbor_shifts):
        return self._compute_impl(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

    def _compute_single_system(
        self, positions, charges, cell, neighbor_indices, neighbor_shifts
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
    calculator = CalculatorTest()
    method = getattr(calculator, method_name)

    result = method(
        positions=positions,
        charges=charges,
        cell=None,
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
    calculator = CalculatorTest()
    match = r"Got inconsistent numbers of positions \(2\) and cell \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[POSITIONS_1, POSITIONS_2],
            charges=[CHARGES_1, CHARGES_2],
            cell=[CELL_1, CELL_2, torch.eye(3)],
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_mismatched_numbers_charges():
    calculator = CalculatorTest()
    match = r"Got inconsistent numbers of positions \(2\) and charges \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[POSITIONS_1, POSITIONS_2],
            charges=[CHARGES_1, CHARGES_2, CHARGES_2],
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_mismatched_numbers_neighbor_indices():
    calculator = CalculatorTest()
    match = r"Got inconsistent numbers of positions \(2\) and neighbor_indices \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[POSITIONS_1, POSITIONS_2],
            charges=[CHARGES_1, CHARGES_2],
            cell=None,
            neighbor_indices=[CHARGES_1, CHARGES_2, POSITIONS_1],
            neighbor_shifts=None,
        )


def test_mismatched_numbers_neighbor_shiftss():
    calculator = CalculatorTest()
    match = r"Got inconsistent numbers of positions \(2\) and neighbor_shifts \(3\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[POSITIONS_1, POSITIONS_2],
            charges=[CHARGES_1, CHARGES_2],
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=[CHARGES_1, CHARGES_2, POSITIONS_1],
        )


# Tests for invalid shape, dtype and device of positions
def test_invalid_shape_positions():
    calculator = CalculatorTest()
    match = (
        r"each `positions` must be a tensor with shape \[n_atoms, 3\], got at least "
        r"one tensor with shape \[4, 5\]"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=torch.ones((4, 5), dtype=DTYPE, device=DEVICE),
            charges=CHARGES_1,
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=None,
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
        calculator.compute(
            positions=[POSITIONS_1, positions_2_wrong_dtype],
            charges=[CHARGES_1, CHARGES_2],
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_device_positions():
    calculator = CalculatorTest()
    match = (
        r"each `positions` must be on the same device cpu as the "
        r"first provided one. Got at least one tensor on device "
        r"meta"
    )
    positions_2_wrong_device = torch.ones((5, 3), dtype=DTYPE, device="meta")
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=[POSITIONS_1, positions_2_wrong_device],
            charges=[CHARGES_1, CHARGES_2],
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


# Tests for invalid shape, dtype and device of cell
def test_invalid_shape_cell():
    calculator = CalculatorTest()
    match = (
        r"each `cell` must be a tensor with shape \[3, 3\], got at least one tensor "
        r"with shape \[2, 2\]"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones([2, 2], dtype=DTYPE, device=DEVICE),
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_dtype_cell():
    calculator = CalculatorTest()
    match = (
        r"each `cell` must have the same type torch.float32 as positions, "
        r"got at least one tensor of type torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones([3, 3], dtype=torch.float64, device=DEVICE),
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_device_cell():
    calculator = CalculatorTest()
    match = (
        r"each `cell` must be on the same device cpu as positions, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones([3, 3], dtype=DTYPE, device="meta"),
            neighbor_indices=None,
            neighbor_shifts=None,
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
        calculator.compute(
            positions=POSITIONS_1,
            charges=torch.ones(len(POSITIONS_1), dtype=DTYPE, device=DEVICE),
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=None,
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
        calculator.compute(
            positions=POSITIONS_1,
            charges=torch.ones((6, 2), dtype=DTYPE, device=DEVICE),
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_dtype_charges():
    calculator = CalculatorTest()
    match = (
        r"each `charges` must have the same type torch.float32 as positions, "
        r"got at least one tensor of type torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=torch.ones((4, 2), dtype=torch.float64, device=DEVICE),
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_device_charges():
    calculator = CalculatorTest()
    match = (
        r"each `charges` must be on the same device cpu as positions, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=torch.ones((4, 2), dtype=DTYPE, device="meta"),
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_cell_no_shifts():
    calculator = CalculatorTest()
    match = r"Provided `cell` but no `neighbor_shifts`."
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones([3, 3], dtype=DTYPE, device=DEVICE),
            neighbor_indices=torch.ones((2, 10), dtype=DTYPE, device=DEVICE),
            neighbor_shifts=None,
        )


def test_shifts_no_cell():
    calculator = CalculatorTest()
    match = r"Provided `neighbor_shifts` but no `cell`."
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=None,
            neighbor_indices=None,
            neighbor_shifts=torch.ones((10, 3), dtype=DTYPE, device=DEVICE),
        )


def test_invalid_shape_neighbor_indices():
    calculator = CalculatorTest()
    match = (
        r"neighbor_indices is expected to have shape \[2, num_neighbors\]"
        r", but got \[4, 10\] for one structure"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=None,
            neighbor_indices=torch.ones((4, 10), dtype=DTYPE, device=DEVICE),
            neighbor_shifts=torch.ones((10, 3), dtype=DTYPE, device=DEVICE),
        )


def test_invalid_shape_neighbor_shifts():
    calculator = CalculatorTest()
    match = (
        r"neighbor_shifts is expected to have shape \[num_neighbors, 3\]"
        r", but got \[10, 2\] for one structure"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones(3, 3, dtype=DTYPE, device=DEVICE),
            neighbor_indices=torch.ones((2, 10), dtype=DTYPE, device=DEVICE),
            neighbor_shifts=torch.ones((10, 2), dtype=DTYPE, device=DEVICE),
        )


def test_invalid_shape_neighbor_indices_neighbor_shifts():
    calculator = CalculatorTest()
    match = (
        r"`neighbor_indices` and `neighbor_shifts` need to have shapes "
        r"\[2, num_neighbors\] and \[num_neighbors, 3\]. For at least one "
        r"structure, got \[2, 10\] and "
        r"\[11, 3\], which is inconsistent"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones(3, 3, dtype=DTYPE, device=DEVICE),
            neighbor_indices=torch.ones((2, 10), dtype=DTYPE, device=DEVICE),
            neighbor_shifts=torch.ones((11, 3), dtype=DTYPE, device=DEVICE),
        )


def test_invalid_device_neighbor_indices():
    calculator = CalculatorTest()
    match = (
        r"each `neighbor_indices` must be on the same device cpu as positions, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=None,
            neighbor_indices=torch.ones((2, 10), dtype=DTYPE, device="meta"),
            neighbor_shifts=torch.ones((10, 3), dtype=DTYPE, device=DEVICE),
        )


def test_invalid_device_neighbor_shifts():
    calculator = CalculatorTest()
    match = (
        r"each `neighbor_shifts` must be on the same device cpu as positions, "
        r"got at least one tensor with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            positions=POSITIONS_1,
            charges=CHARGES_1,
            cell=torch.ones([3, 3], dtype=DTYPE, device=DEVICE),
            neighbor_indices=torch.ones((2, 10), dtype=DTYPE, device=DEVICE),
            neighbor_shifts=torch.ones((10, 3), dtype=DTYPE, device="meta"),
        )


def test_exponent_out_of_range():
    match = r"`exponent` p=.* has to satisfy 0 < p <= 3"
    with pytest.raises(ValueError, match=match):
        PeriodicBase(exponent=-1, atomic_smearing=0.1, subtract_interior=True)
    with pytest.raises(ValueError, match=match):
        PeriodicBase(exponent=4, atomic_smearing=0.1, subtract_interior=True)


def test_atomic_smearing_non_positive():
    match = r"`atomic_smearing` .* has to be positive"
    with pytest.raises(ValueError, match=match):
        PeriodicBase(exponent=2, atomic_smearing=0, subtract_interior=True)
    with pytest.raises(ValueError, match=match):
        PeriodicBase(exponent=2, atomic_smearing=-0.1, subtract_interior=True)


def periodic_base():
    return PeriodicBase(exponent=2, atomic_smearing=0.1, subtract_interior=True)


def test_prepare_no_cell():
    match = r"provide `cell` for periodic calculation"
    with pytest.raises(ValueError, match=match):
        periodic_base()._prepare(None, torch.tensor([0]), torch.tensor([0]))


def test_prepare_no_neighbor_indices():
    match = r"provide `neighbor_indices` for periodic calculation"
    with pytest.raises(ValueError, match=match):
        periodic_base()._prepare(torch.tensor([0]), None, torch.tensor([0]))


def test_prepare_no_neighbor_shifts():
    match = r"provide `neighbor_shifts` for periodic calculation"
    with pytest.raises(ValueError, match=match):
        periodic_base()._prepare(torch.tensor([0]), torch.tensor([0]), None)
