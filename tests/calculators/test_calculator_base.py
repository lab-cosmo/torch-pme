import pytest
import torch

from meshlode.calculators.base import CalculatorBase


class TestCalculator(CalculatorBase):
    def compute(
        self, types, positions, cell, charges, neighbor_indices, neighbor_shifts
    ):
        return self._compute_impl(
            types=types,
            positions=positions,
            cell=cell,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

    def forward(
        self, types, positions, cell, charges, neighbor_indices, neighbor_shifts
    ):
        return self._compute_impl(
            types=types,
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
    "types, positions, charges",
    [
        (torch.arange(2), torch.ones([2, 3]), torch.ones(2)),
        ([torch.arange(2)], [torch.ones([2, 3])], [torch.ones(2)]),
        (
            [torch.arange(2), torch.arange(4)],
            [torch.ones([2, 3]), torch.ones([4, 3])],
            [torch.ones(2), torch.ones(4)],
        ),
    ],
)
def test_compute(method_name, types, positions, charges):
    calculator = TestCalculator(all_types=None, exponent=1.0)
    method = getattr(calculator, method_name)

    result = method(
        types=types,
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


def test_mismatched_lengths_types_positions():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = r"inconsistent lengths of types \(\d+\) positions \(\d+\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2),
            positions=[torch.ones([2, 3]), torch.ones([3, 3])],
            cell=None,
            charges=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_shape_positions():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = (
        r"each `positions` must be a \(n_types x 3\) tensor, got at least one tensor "
        r"with shape \[3, 3\]"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2),
            positions=torch.ones([3, 3]),
            cell=None,
            charges=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_mismatched_lengths_types_cell():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = r"inconsistent lengths of types \(\d+\) and cell \(\d+\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2),
            positions=torch.ones([2, 3]),
            cell=[torch.ones([3, 3]), torch.ones([3, 3])],
            charges=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_inconsistent_devices():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = r"Inconsistent devices of types \([a-zA-Z:]+\) and positions \([a-zA-Z:]+\)"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2, device="meta"),
            positions=torch.ones([2, 3], device="cpu"),
            cell=None,
            charges=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_inconsistent_dtypes_cell():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = (
        r"`cell` must be have the same dtype as `positions`, got "
        r"torch.float32 and torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2),
            positions=torch.ones([2, 3], dtype=torch.float64),
            cell=torch.ones([3, 3], dtype=torch.float32),
            charges=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_inconsistent_dtypes_charges():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = (
        r"`charges` must be have the same dtype as `positions`, got "
        r"torch.float32 and torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2),
            positions=torch.ones([2, 3], dtype=torch.float64),
            cell=None,
            charges=torch.ones([2], dtype=torch.float32),
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_mismatched_lengths_types_charges():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = (
        r"The first dimension of `charges` must be the same as the length of `types`, "
        r"got \d+ and \d+"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2),
            positions=torch.ones([2, 3]),
            cell=None,
            charges=torch.ones([3]),
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_shape_cell():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = (
        r"each `cell` must be a \(3 x 3\) tensor, got at least one tensor with "
        r"shape \[2, 2\]"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2),
            positions=torch.ones([2, 3]),
            cell=torch.ones([2, 2]),
            charges=None,
            neighbor_indices=None,
            neighbor_shifts=None,
        )


def test_invalid_shape_neighbor_indices():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = r"Expected shape of neighbor_indices is \(2, \d+\), but got \[\d+, \d+\]"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2),
            positions=torch.ones([2, 3]),
            cell=None,
            charges=None,
            neighbor_indices=torch.ones([3, 2]),
            neighbor_shifts=None,
        )


def test_invalid_shape_neighbor_shifts():
    calculator = TestCalculator(all_types=None, exponent=1.0)
    match = r"Expected shape of neighbor_shifts is \(3, \d+\), but got \[\d+, \d+\]"
    with pytest.raises(ValueError, match=match):
        calculator.compute(
            types=torch.arange(2),
            positions=torch.ones([2, 3]),
            cell=None,
            charges=None,
            neighbor_indices=None,
            neighbor_shifts=torch.ones([3, 3]),
        )
