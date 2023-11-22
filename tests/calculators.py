import torch
from packaging import version

from meshlode import calculators
from meshlode.system import System


def system():
    return System(
        species=torch.tensor([1, 1, 8, 8]),
        positions=torch.tensor([[0.0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        cell=torch.tensor([[10, 0, 0], [0, 10, 0], [0, 0, 10]]),
    )


def spherical_expansion():
    return calculators.MeshPotential(
        atomic_gaussian_width=1,
    )


def check_operation(calculator):
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type

    descriptor = calculator.compute(system(), gradients=["positions"])

    assert isinstance(descriptor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert descriptor._type().name() == "TensorMap"


def test_operation_as_python():
    check_operation(spherical_expansion())


def test_operation_as_torch_script():
    scripted = torch.jit.script(spherical_expansion())
    check_operation(scripted)
