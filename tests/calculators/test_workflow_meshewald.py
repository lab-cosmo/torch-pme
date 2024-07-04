"""Basic tests if the calculator works and is torch scriptable. Actual tests are done
for the metatensor calculator."""

import math

import pytest
import torch
from torch.testing import assert_close

from meshlode import MeshEwaldPotential, MeshPotential
from meshlode.calculators.calculator_base import _1d_tolist, _is_subset


MADELUNG_CSCL = torch.tensor(2 * 1.7626 / math.sqrt(3))
CHARGES_CSCL = torch.tensor([1.0, -1.0])


def cscl_system():
    """CsCl crystal. Same as in the madelung test"""
    types = torch.tensor([55, 17])
    positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    cell = torch.eye(3)

    return types, positions, cell


def cscl_system_with_charges():
    """CsCl crystal with charges."""
    charges = torch.tensor([[0.0, 1.0], [1.0, 0]])
    return cscl_system() + (charges,)


# Initialize the calculators. For now, only the MeshPotential is implemented.
def descriptor() -> MeshEwaldPotential:
    atomic_smearing = 0.1
    return MeshEwaldPotential(
        atomic_smearing=atomic_smearing,
        mesh_spacing=atomic_smearing / 4,
        interpolation_order=2,
        subtract_self=True,
    )


def test_forward():
    mp = descriptor()
    descriptor_compute = mp.compute(*cscl_system())
    descriptor_forward = mp.forward(*cscl_system())

    assert torch.equal(descriptor_forward, descriptor_compute)


def test_atomic_smearing_error():
    with pytest.raises(ValueError, match="has to be positive"):
        MeshEwaldPotential(atomic_smearing=-1.0)


def test_interpolation_order_error():
    with pytest.raises(ValueError, match="Only `interpolation_order` from 1 to 5"):
        MeshEwaldPotential(atomic_smearing=1, interpolation_order=10)


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
    types, pos, cell = cscl_system()
    print(cell)
    descriptor = calculator.compute(types=types, positions=pos, cell=cell)
    assert type(descriptor) is torch.Tensor


# Run the above test as a normal python script
def test_operation_as_python():
    check_operation(descriptor())


"""
# Similar to the above, but also testing that the code can be compiled as a torch script
# Disabled for now since (1) the ASE neighbor list and (2) the use of the potential
# class are clashing with the torch script capabilities.
def test_operation_as_torch_script():
    scripted = torch.jit.script(descriptor())
    check_operation(scripted)
"""


def test_single_frame():
    values = descriptor().compute(*cscl_system())
    assert_close(
        MADELUNG_CSCL,
        CHARGES_CSCL[0] * values[0, 0] + CHARGES_CSCL[1] * values[0, 1],
        atol=1e4,
        rtol=1e-5,
    )


# Test with explicit charges
def test_single_frame_with_charges():
    values = descriptor().compute(*cscl_system_with_charges())
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


def test_charges_error_dimension_mismatch():
    types = torch.tensor([1, 2])
    positions = torch.zeros((2, 3))
    cell = torch.eye(3)
    charges = torch.zeros((1, 2))  # This should have the same first dimension as types

    match = (
        "The first dimension of `charges` must be the same as the length "
        "of `types`, got 1 and 2."
    )

    with pytest.raises(ValueError, match=match):
        descriptor().compute(
            types=types, positions=positions, cell=cell, charges=charges
        )


def test_charges_error_length_mismatch():
    types = [torch.tensor([1, 2]), torch.tensor([1, 2, 3])]
    positions = [torch.zeros((2, 3)), torch.zeros((3, 3))]
    cell = [torch.eye(3), torch.eye(3)]
    charges = [torch.zeros(2, 1)]  # This should have the same length as types
    match = "The number of `types` and `charges` tensors must be the same, got 2 and 1."

    with pytest.raises(ValueError, match=match):
        descriptor().compute(
            types=types, positions=positions, cell=cell, charges=charges
        )


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


def test_dtype_device():
    """Test that the output dtype and device are the same as the input."""
    device = "cpu"
    dtype = torch.float64

    types = torch.tensor([1], dtype=dtype, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device)

    MP = MeshPotential(atomic_smearing=0.2)
    potential = MP.compute(types=types, positions=positions, cell=cell)

    assert potential.dtype == dtype
    assert potential.device.type == device


def test_inconsistent_dtype():
    """Test if the cell and positions have inconsistent dtype and error is raised."""
    types = torch.tensor([1], dtype=torch.float32)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)  # Different dtype
    cell = torch.eye(3, dtype=torch.float32)

    MP = MeshPotential(atomic_smearing=0.2)

    match = (
        "`cell` must be have the same dtype as `positions`, got torch.float32 and "
        "torch.float64"
    )
    with pytest.raises(ValueError, match=match):
        MP.compute(types=types, positions=positions, cell=cell)


def test_inconsistent_device():
    """Test if the cell and positions have inconsistent device and error is raised."""
    types = torch.tensor([1], device="cpu")
    positions = torch.tensor([[0.0, 0.0, 0.0]], device="cpu")
    cell = torch.eye(3, device="meta")  # different device

    MP = MeshPotential(atomic_smearing=0.2)

    match = r"Inconsistent devices of types \(cpu\) and cell \(meta\)"
    with pytest.raises(ValueError, match=match):
        MP.compute(types=types, positions=positions, cell=cell)


def test_inconsistent_device_charges():
    """Test if the cell and positions have inconsistent device and error is raised."""
    types = torch.tensor([1], device="cpu")
    positions = torch.tensor([[0.0, 0.0, 0.0]], device="cpu")
    cell = torch.eye(3, device="cpu")
    charges = torch.tensor([0.0], device="meta")  # different device

    MP = MeshPotential(atomic_smearing=0.2)

    match = "`charges` must be on the same device as `positions`, got meta and cpu."
    with pytest.raises(ValueError, match=match):
        MP.compute(types=types, positions=positions, cell=cell, charges=charges)


def test_inconsistent_dtype_charges():
    """Test if the cell and positions have inconsistent dtype and error is raised."""
    types = torch.tensor([1], dtype=torch.float32)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32)
    charges = torch.tensor([0.0], dtype=torch.float64)  # Different dtype

    MP = MeshPotential(atomic_smearing=0.2)

    match = (
        "`charges` must be have the same dtype as `positions`, got torch.float64 and "
        "torch.float32"
    )
    with pytest.raises(ValueError, match=match):
        MP.compute(types=types, positions=positions, cell=cell, charges=charges)


def test_1d_tolist():
    in_list = [1, 2, 7, 3, 4, 42]
    in_tensor = torch.tensor(in_list)
    assert _1d_tolist(in_tensor) == in_list


def test_is_subset_true():
    subset_candidate = [1, 2]
    superset = [1, 2, 3, 4, 5]
    assert _is_subset(subset_candidate, superset)


def test_is_subset_false():
    subset_candidate = [1, 2, 8]
    superset = [1, 2, 3, 4, 5]
    assert not _is_subset(subset_candidate, superset)
