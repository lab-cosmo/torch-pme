import pytest
import torch
from packaging import version

import torchpme
import torchpme.calculators

mts_torch = pytest.importorskip("metatensor.torch")
mta_torch = pytest.importorskip("metatomic.torch")


@pytest.fixture
def system():
    system = mta_torch.System(
        types=torch.tensor([1, 2, 2]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.5]]),
        cell=4.2 * torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )

    charges = torch.tensor([1.0, -0.5, -0.5]).unsqueeze(1)
    block = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    tensor = mts_torch.TensorMap(
        keys=mts_torch.Labels("_", torch.zeros(1, 1, dtype=torch.int32)), blocks=[block]
    )

    system.add_data(name="charges", tensor=tensor)

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
        r"dtype of `neighbors` \(torch.float64\) must be the same as `system` "
        r"\(torch.float32\)"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(system, neighbors)


def test_wrong_neighbors_device(system, neighbors):
    neighbors = neighbors.to("meta")

    calculator = CalculatorTest()
    match = r"device of `neighbors` \(meta\) must be the same as `system` \(cpu\)"
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
    system_nocharge = mta_torch.System(
        system.types, system.positions, system.cell, pbc=system.pbc
    )

    calculator = CalculatorTest()

    match = r"`system` does not contain `charges` data"
    with pytest.raises(ValueError, match=match):
        calculator.forward(system_nocharge, neighbors)


def test_different_number_charge_channels(system, neighbors):
    charges = torch.zeros(2, 2)

    block = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    tensor = mts_torch.TensorMap(
        keys=mts_torch.Labels("_", torch.zeros(1, 1, dtype=torch.int32)),
        blocks=[block],
    )

    system.add_data(name="charges", tensor=tensor, override=True)

    calculator = CalculatorTest()

    match = (
        r"`charges` must be a tensor with shape \[n_atoms, n_channels\], with "
        r"`n_atoms` being the same as the variable `positions`. Got tensor with "
        r"shape \[2, 2\] where positions contains 3 atoms"
    )
    with pytest.raises(ValueError, match=match):
        calculator.forward(system, neighbors)


def test_2_blocks_error(system, neighbors):
    """Test that a charges TensorMap with 2 blocks raises an error."""
    charges = torch.zeros(2, 2)

    block = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    tensor = mts_torch.TensorMap(
        keys=mts_torch.Labels("_", torch.arange(2, dtype=torch.int32).reshape(-1, 1)),
        blocks=[block, block],
    )

    system.add_data(name="charges", tensor=tensor, override=True)

    calculator = CalculatorTest()

    match = "Charge tensor have exactlty one block but has 2 blocks"
    with pytest.raises(ValueError, match=match):
        calculator.forward(system, neighbors)


def test_components_error(system, neighbors):
    """Test that a charge block containing components raises an error."""
    charges = torch.zeros(2, 1, 2)

    single_label = mts_torch.Labels("_", torch.zeros(1, 1, dtype=torch.int32))

    block = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[single_label],
        properties=mts_torch.Labels.range("charge", charges.shape[-1]),
    )

    tensor = mts_torch.TensorMap(keys=single_label, blocks=[block])
    system.add_data(name="charges", tensor=tensor, override=True)

    calculator = CalculatorTest()

    match = "TensorBlock containg the charges should not have components; found 1"
    with pytest.raises(ValueError, match=match):
        calculator.forward(system, neighbors)


def test_systems_with_different_number_of_atoms(system, neighbors):
    """Test that systems with different numnber of atoms are supported."""
    system_more_atoms = mta_torch.System(
        types=torch.tensor([1, 1, 8]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 2.0]]),
        cell=torch.zeros([3, 3]),
        pbc=torch.tensor([True, True, True]),
    )

    charges = torch.tensor([1.0, -1.0, 2.0]).unsqueeze(1)
    block = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    tensor = mts_torch.TensorMap(
        keys=mts_torch.Labels("_", torch.zeros(1, 1, dtype=torch.int32)), blocks=[block]
    )

    system_more_atoms.add_data(name="charges", tensor=tensor)

    calculator = CalculatorTest()
    calculator.forward(system, neighbors)
    calculator.forward(system_more_atoms, neighbors)
