import pytest
import torch
from packaging import version


mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")
meshlode_metatensor = pytest.importorskip("meshlode.metatensor")


class CalculatorTest(meshlode_metatensor.base.CalculatorBaseMetatensor):
    def _compute_single_system(
        self, positions, charges, cell, neighbor_indices, neighbor_shifts
    ):
        return charges


@pytest.mark.parametrize("method_name", ["compute", "forward"])
def test_compute_output_shapes_single(method_name):
    system = mts_atomistic.System(
        types=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        cell=torch.zeros([3, 3]),
    )

    charges = torch.tensor([1.0, -1.0]).reshape(-1, 1)
    data = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    system.add_data(name="charges", data=data)

    calculator = CalculatorTest()
    method = getattr(calculator, method_name)
    result = method(system)

    assert isinstance(result, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert result._type().name() == "TensorMap"

    assert len(result) == 1
    assert result[0].samples.names == ["system", "atom"]
    assert result[0].components == []
    assert result[0].properties.names == ["charges_channel"]

    assert tuple(result[0].values.shape) == (len(system), 1)


def test_compute_output_shapes_multiple():

    system = mts_atomistic.System(
        types=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        cell=torch.zeros([3, 3]),
    )

    charges = torch.tensor([1.0, -1.0]).reshape(-1, 1)
    data = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    system.add_data(name="charges", data=data)

    calculator = CalculatorTest()
    result = calculator.compute([system, system])

    assert isinstance(result, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert result._type().name() == "TensorMap"

    assert len(result) == 1
    assert result[0].samples.names == ["system", "atom"]
    assert result[0].components == []
    assert result[0].properties.names == ["charges_channel"]

    assert tuple(result[0].values.shape) == (2 * len(system), 1)


def test_wrong_system_dtype():
    system1 = mts_atomistic.System(
        types=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        cell=torch.zeros([3, 3]),
    )

    system2 = mts_atomistic.System(
        types=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64),
        cell=torch.zeros([3, 3], dtype=torch.float64),
    )

    calculator = CalculatorTest()

    match = r"`dtype` of all systems must be the same, got 7 and 6"
    with pytest.raises(ValueError, match=match):
        calculator.compute([system1, system2])


def test_wrong_system_device():
    system1 = mts_atomistic.System(
        types=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        cell=torch.zeros([3, 3]),
    )

    system2 = mts_atomistic.System(
        types=torch.tensor([1, 1], device="meta"),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], device="meta"),
        cell=torch.zeros([3, 3], device="meta"),
    )

    calculator = CalculatorTest()

    match = r"`device` of all systems must be the same, got meta and cpu"
    with pytest.raises(ValueError, match=match):
        calculator.compute([system1, system2])


def test_wrong_system_not_all_charges():
    system1 = mts_atomistic.System(
        types=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        cell=torch.zeros([3, 3]),
    )

    charges = torch.tensor([1.0, -1.0]).reshape(-1, 1)
    data = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    system1.add_data(name="charges", data=data)

    system2 = mts_atomistic.System(
        types=torch.tensor(
            [1, 1],
        ),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        cell=torch.zeros([3, 3]),
    )

    calculator = CalculatorTest()

    match = r"`systems` do not consistently contain `charges` data"
    with pytest.raises(ValueError, match=match):
        calculator.compute([system1, system2])


def test_different_number_charge_channles():
    system1 = mts_atomistic.System(
        types=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        cell=torch.zeros([3, 3]),
    )

    charges1 = torch.tensor([1.0, -1.0]).reshape(-1, 1)
    data1 = mts_torch.TensorBlock(
        values=charges1,
        samples=mts_torch.Labels.range("atom", charges1.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges1.shape[1]),
    )

    system1.add_data(name="charges", data=data1)

    system2 = mts_atomistic.System(
        types=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        cell=torch.zeros([3, 3]),
    )

    charges2 = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
    data2 = mts_torch.TensorBlock(
        values=charges2,
        samples=mts_torch.Labels.range("atom", charges2.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges2.shape[1]),
    )
    system2.add_data(name="charges", data=data2)

    calculator = CalculatorTest()

    match = (
        r"number of charges-channels in system index 1 \(2\) is inconsistent with "
        r"first system \(1\)"
    )
    with pytest.raises(ValueError, match=match):
        calculator.compute([system1, system2])
