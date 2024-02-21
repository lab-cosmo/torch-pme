"""Basic tests if the calculator works and is torch scriptable. Actual tests are done
for the metatensor calculator."""

import math

import torch
from torch.testing import assert_close

from meshlode import MeshPotential, System


MADELUNG_CSCL = torch.tensor(2 * 1.7626 / math.sqrt(3))
CHARGES_CSCL = torch.tensor([1.0, -1.0])


def cscl_system() -> System:
    """CsCl crystal. Same as in the madelung test"""
    return System(
        species=torch.tensor([55, 17]),
        positions=torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]]),
        cell=torch.eye(3),
    )


# Initialize the calculators. For now, only the MeshPotential is implemented.
def descriptor() -> MeshPotential:
    atomic_smearing = 0.1
    return MeshPotential(
        atomic_smearing=atomic_smearing,
        mesh_spacing=atomic_smearing / 4,
        interpolation_order=2,
        subtract_self=True,
    )


# Make sure that the calculators are computing the features without raising errors,
# and returns the correct output format (TensorMap)
def check_operation(calculator):
    descriptor = calculator.compute(cscl_system())
    assert type(descriptor) is torch.Tensor


# Run the above test as a normal python script
def test_operation_as_python():
    check_operation(descriptor())


# Similar to the above, but also testing that the code can be compiled as a torch script
def test_operation_as_torch_script():
    scripted = torch.jit.script(descriptor())
    check_operation(scripted)


def test_single_frame():
    values = descriptor().compute(cscl_system())
    print(values)
    assert_close(
        MADELUNG_CSCL,
        CHARGES_CSCL[0] * values[0, 0] + CHARGES_CSCL[1] * values[0, 1],
        atol=1e4,
        rtol=1e-5,
    )


def test_multi_frame():
    l_values = descriptor().compute([cscl_system(), cscl_system()])
    for values in l_values:
        assert_close(
            MADELUNG_CSCL,
            CHARGES_CSCL[0] * values[0, 0] + CHARGES_CSCL[1] * values[0, 1],
            atol=1e4,
            rtol=1e-5,
        )
