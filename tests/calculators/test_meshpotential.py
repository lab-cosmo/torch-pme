"""Basic tests if the calculator works and is torch scriptable. Actual tests are done
for the metatensor calculator."""

import math

import pytest
import torch
from torch.testing import assert_close

from meshlode import MeshPotential


MADELUNG_CSCL = torch.tensor(2 * 1.7626 / math.sqrt(3))
CHARGES_CSCL = torch.tensor([1.0, -1.0])


def cscl_system():
    """CsCl crystal. Same as in the madelung test"""
    types = torch.tensor([55, 17])
    positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    cell = torch.eye(3)

    return types, positions, cell


# Initialize the calculators. For now, only the MeshPotential is implemented.
def descriptor() -> MeshPotential:
    atomic_smearing = 0.1
    return MeshPotential(
        atomic_smearing=atomic_smearing,
        mesh_spacing=atomic_smearing / 4,
        interpolation_order=2,
        subtract_self=True,
    )


def test_atomic_smearing_error():
    with pytest.raises(ValueError, match="has to be positive"):
        MeshPotential(atomic_smearing=-1.0)


def test_interpolation_order_error():
    with pytest.raises(ValueError, match="Only `interpolation_order` from 1 to 5"):
        MeshPotential(atomic_smearing=1, interpolation_order=10)


def test_all_types():
    descriptor = MeshPotential(atomic_smearing=0.1, all_types=[8, 55, 17])
    values = descriptor.compute(*cscl_system())
    assert values.shape == (2, 3)
    assert torch.equal(values[:, 0], torch.zeros(2))


def test_all_types_error():
    descriptor = MeshPotential(atomic_smearing=0.1, all_types=[17])
    with pytest.raises(ValueError, match="Global list of types"):
        descriptor.compute(*cscl_system())


# Make sure that the calculators are computing the features without raising errors,
# and returns the correct output format (TensorMap)
def check_operation(calculator):
    descriptor = calculator.compute(*cscl_system())
    assert type(descriptor) is torch.Tensor


# Run the above test as a normal python script
def test_operation_as_python():
    check_operation(descriptor())


# Similar to the above, but also testing that the code can be compiled as a torch script
def test_operation_as_torch_script():
    scripted = torch.jit.script(descriptor())
    check_operation(scripted)


def test_single_frame():
    values = descriptor().compute(*cscl_system())
    print(values)
    assert_close(
        MADELUNG_CSCL,
        CHARGES_CSCL[0] * values[0, 0] + CHARGES_CSCL[1] * values[0, 1],
        atol=1e4,
        rtol=1e-5,
    )


def test_multi_frame():
    types, positions, cell = cscl_system()
    l_values = descriptor().compute(
        types=[types, types], positions=[positions, positions], cell=[cell, cell]
    )
    for values in l_values:
        assert_close(
            MADELUNG_CSCL,
            CHARGES_CSCL[0] * values[0, 0] + CHARGES_CSCL[1] * values[0, 1],
            atol=1e4,
            rtol=1e-5,
        )


def test_types_error():
    types = torch.tensor([[1, 2], [3, 4]])  # This is a 2D tensor, should be 1D
    positions = torch.zeros((2, 3))
    cell = torch.eye(3)

    match = (
        "each `types` must be a 1 dimensional tensor, got at least one tensor with "
        "2 dimensions"
    )
    with pytest.raises(ValueError, match=match):
        descriptor().compute(types=types, positions=positions, cell=cell)


def test_positions_error():
    types = torch.tensor([1, 2])
    positions = torch.zeros(
        (1, 3)
    )  # This should have the same first dimension as types
    cell = torch.eye(3)

    match = (
        "each `positions` must be a \\(n_types x 3\\) tensor, got at least "
        "one tensor with shape \\[1, 3\\]"
    )

    with pytest.raises(ValueError, match=match):
        descriptor().compute(types=types, positions=positions, cell=cell)


def test_cell_error():
    types = torch.tensor([1, 2, 3])
    positions = torch.zeros((3, 3))
    cell = torch.eye(2)  # This is a 2x2 tensor, should be 3x3

    match = (
        "each `cell` must be a \\(3 x 3\\) tensor, got at least one tensor "
        "with shape \\[2, 2\\]"
    )
    with pytest.raises(ValueError, match=match):
        descriptor().compute(types=types, positions=positions, cell=cell)


def test_positions_cell_dtype_error():
    types = torch.tensor([1, 2, 3])
    positions = torch.zeros((3, 3), dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float64)

    match = (
        "`cell` must be have the same dtype as `positions`, got torch.float64 "
        "and torch.float32"
    )
    with pytest.raises(ValueError, match=match):
        descriptor().compute(types=types, positions=positions, cell=cell)
