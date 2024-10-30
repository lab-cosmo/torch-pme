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
        pbc=torch.tensor([True, True, True]),
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


# non-range-separated Coulomb direct calculator
class CalculatorTestTorch(torchpme.Calculator):
    def __init__(self, potential=None):
        super().__init__(
            potential=potential
            or torchpme.CoulombPotential(smearing=None, exclusion_radius=None)
        )


class CalculatorTest(torchpme.metatensor.Calculator):
    _base_calculator = CalculatorTestTorch

    def __init__(self):
        super().__init__(potential=torchpme.CoulombPotential(smearing=None))


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


def test_wrong_neighbors_dtype(system, neighbors):
    system = system.to(torch.float32)
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
        system.types, system.positions, system.cell, pbc=system.pbc
    )

    calculator = CalculatorTest()

    match = r"`system` does not contain `charges` data"
    with pytest.raises(ValueError, match=match):
        calculator.forward(system_nocharge, neighbors)


def test_different_number_charge_channels(system, neighbors):
    system_channels = mts_atomistic.System(
        system.types, system.positions, system.cell, pbc=system.pbc
    )

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
        r"each `charges` must be a tensor with shape \[n_atoms, n_channels\], "
        r"with `n_atoms` being the same as the variable `positions`. "
        r"Got at least one tensor with shape \[2, 2\] where positions contains 3 atoms"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(system_channels, neighbors)


def test_systems_with_different_number_of_atoms(system, neighbors):
    """Test that systems with different numnber of atoms are supported."""
    system_more_atoms = mts_atomistic.System(
        types=torch.tensor([1, 1, 8]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 2.0]]),
        cell=torch.zeros([3, 3]),
        pbc=torch.tensor([True, True, True]),
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
    calculator.forward(system, neighbors)
    calculator.forward(system_more_atoms, neighbors)
