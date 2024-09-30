import pytest
import torch
from packaging import version

import torchpme
import torchpme.calculators

mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")


@pytest.fixture
def system():
    system = mts_atomistic.System(
        types=torch.tensor([1, 2, 2]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.5]]),
        cell=4.2 * torch.eye(3),
    )

    charges = torch.tensor([1.0, -0.5, -0.5]).unsqueeze(1)
    data = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    system.add_data(name="charges", data=data)

    return system


@pytest.fixture
def neighbors():
    n_neighbors = 1
    sample_values = torch.zeros(n_neighbors, 5, dtype=torch.int32)
    samples = mts_torch.Labels(
        names=[
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=sample_values,
    )

    values = torch.zeros(n_neighbors, 3, 1)

    return mts_torch.TensorBlock(
        values=values,
        samples=samples,
        components=[mts_torch.Labels.range("xyz", 3)],
        properties=mts_torch.Labels.range("distance", 1),
    )


class CalculatorTestTorch(torchpme.calculators.base.CalculatorBaseTorch):
    def _compute_single_system(
        self, positions, charges, cell, neighbor_indices, neighbor_distances
    ):
        self._positions = positions
        self._charges = charges
        self._cell = cell
        self._neighbor_indices = neighbor_indices
        self._neighbor_distances = neighbor_distances

        return charges


class CalculatorTest(torchpme.metatensor.base.CalculatorBaseMetatensor):
    def __init__(self):
        super().__init__()
        self.calculator = CalculatorTestTorch(
            potential=torchpme.lib.InversePowerLawPotential(exponent=1.0, smearing=0.0)
        )


def test_compute_output_shapes_single(system, neighbors):
    calculator = CalculatorTest()
    result = calculator.forward(system, neighbors)

    assert isinstance(result, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert result._type().name() == "TensorMap"

    assert len(result) == 1
    assert result[0].samples.names == ["system", "atom"]
    assert result[0].components == []
    assert result[0].properties.names == ["charges_channel"]

    assert tuple(result[0].values.shape) == (len(system), 1)


def test_corrrect_value_extraction_from_neighbors_tensormap(system, neighbors):
    calculator = CalculatorTest()
    calculator.forward(system, neighbors)

    neighbor_indices = neighbors.samples.view(["first_atom", "second_atom"]).values
    neighbor_distances = torch.linalg.norm(neighbors.values, dim=1).squeeze(1)

    assert torch.equal(
        calculator.calculator._charges, torch.tensor([[1.0], [-0.5], [-0.5]])
    )
    assert torch.equal(calculator.calculator._neighbor_indices, neighbor_indices)
    assert torch.equal(calculator.calculator._neighbor_distances, neighbor_distances)


def test_compute_output_shapes_multiple(system, neighbors):
    calculator = CalculatorTest()
    result = calculator.forward([system, system], [neighbors, neighbors])

    assert isinstance(result, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert result._type().name() == "TensorMap"

    assert len(result) == 1
    assert result[0].samples.names == ["system", "atom"]
    assert result[0].components == []
    assert result[0].properties.names == ["charges_channel"]

    assert tuple(result[0].values.shape) == (2 * len(system), 1)


def test_type_check_error(system, neighbors):
    calculator = CalculatorTest()

    match = (
        "Inconsistent parameter types. `systems` is a list, while `neighbors` is a "
        "TensorBlock. Both need either be a list or System/TensorBlock!"
    )
    with pytest.raises(TypeError, match=match):
        calculator._validate_compute_parameters([system], neighbors)

    match = (
        "Inconsistent parameter types. `systems` is a not a list, while `neighbors` "
        "is a list. Both need either be a list or System/TensorBlock!"
    )
    with pytest.raises(TypeError, match=match):
        calculator._validate_compute_parameters(system, [neighbors])


def test_inconsistent_length(system):
    calculator = CalculatorTest()
    match = r"Got inconsistent numbers of systems \(1\) and neighbors \(2\)"
    with pytest.raises(ValueError, match=match):
        calculator.forward(systems=[system], neighbors=[None, None])


def test_wrong_system_dtype(system, neighbors):
    system_float64 = system.to(torch.float64)

    calculator = CalculatorTest()

    match = (
        r"`dtype` of all systems must be the same, got torch.float64 and torch.float32"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward([system, system_float64], [neighbors, neighbors])


def test_wrong_system_device(system, neighbors):
    system_meta = system.to("meta")

    calculator = CalculatorTest()

    match = r"`device` of all systems must be the same, got meta and cpu"
    with pytest.raises(ValueError, match=match):
        calculator.forward([system, system_meta], [neighbors, neighbors])


def test_wrong_neighbors_dtype(system, neighbors):
    neighbors = neighbors.to(torch.float64)

    calculator = CalculatorTest()
    match = (
        "each `neighbors` must have the same type torch.float32 as "
        "`systems`, got at least one `neighbors` of type torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(system, neighbors)


def test_wrong_neighbors_device(system, neighbors):
    neighbors = neighbors.to("meta")

    calculator = CalculatorTest()
    match = (
        "each `neighbors` must be on the same device cpu as "
        "`systems`, got at least one `neighbors` with device meta"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(system, neighbors)


def test_wrong_neighbors_samples(system, neighbors):
    neighbors = mts_torch.TensorBlock(
        values=neighbors.values,
        samples=neighbors.samples.rename("first_atom", "foo"),
        components=neighbors.components,
        properties=neighbors.properties,
    )

    calculator = CalculatorTest()
    match = (
        "Invalid samples for `neighbors`: the sample names must be "
        "'first_atom', 'second_atom', 'cell_shift_a', 'cell_shift_b', "
        "'cell_shift_c'"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(system, neighbors)


def test_wrong_neighbors_components(system, neighbors):
    neighbors = mts_torch.TensorBlock(
        values=neighbors.values,
        samples=neighbors.samples,
        components=[mts_torch.Labels.range("abc", 3)],  # abc instead of xyz
        properties=neighbors.properties,
    )

    calculator = CalculatorTest()
    match = (
        "Invalid components for `neighbors`: there should be a single "
        r"'xyz'=\[0, 1, 2\] component"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(system, neighbors)


def test_wrong_neighbors_properties(system, neighbors):
    neighbors = mts_torch.TensorBlock(
        values=neighbors.values,
        samples=neighbors.samples,
        components=neighbors.components,
        properties=neighbors.properties.rename("distance", "foo"),
    )

    calculator = CalculatorTest()
    match = (
        "Invalid properties for `neighbors`: there should be a single "
        "'distance'=0 property"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(system, neighbors)


def test_wrong_system_not_all_charges(system, neighbors):
    system_nocharge = mts_torch.atomistic.System(
        system.types, system.positions, system.cell
    )

    calculator = CalculatorTest()

    match = r"`systems` do not consistently contain `charges` data"
    with pytest.raises(ValueError, match=match):
        calculator.forward([system, system_nocharge], [neighbors, neighbors])


def test_different_number_charge_channels(system, neighbors):
    system_channels = mts_atomistic.System(system.types, system.positions, system.cell)

    charges2 = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
    data2 = mts_torch.TensorBlock(
        values=charges2,
        samples=mts_torch.Labels.range("atom", charges2.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges2.shape[1]),
    )
    system_channels.add_data(name="charges", data=data2)

    calculator = CalculatorTest()

    match = (
        r"number of charges-channels in system index 1 \(2\) is inconsistent with "
        r"first system \(1\)"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward([system, system_channels], [neighbors, neighbors])


def test_systems_with_different_number_of_atoms(system, neighbors):
    """Test that systems with different numnber of atoms are supported."""
    system_more_atoms = mts_atomistic.System(
        types=torch.tensor([1, 1, 8]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 2.0]]),
        cell=torch.zeros([3, 3]),
    )

    charges = torch.tensor([1.0, -1.0, 2.0]).unsqueeze(1)
    data = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    system_more_atoms.add_data(name="charges", data=data)

    calculator = CalculatorTest()
    calculator.forward([system, system_more_atoms], [neighbors, neighbors])
