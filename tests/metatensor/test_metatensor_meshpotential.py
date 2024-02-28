from typing import List

import pytest
import torch
from packaging import version

from meshlode import System


metatensor_torch = pytest.importorskip("metatensor.torch")
meshlode_metatensor = pytest.importorskip("meshlode.metatensor")


# Define toy system consisting of a single structure for testing
def toy_system_single_frame() -> System:
    return System(
        species=torch.tensor([1, 1, 8, 8]),
        positions=torch.tensor([[0.0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        cell=torch.tensor([[10.0, 0, 0], [0, 10, 0], [0, 0, 10]]),
    )


# Initialize the calculators. For now, only the meshlode_metatensor.MeshPotential is
# implemented.
def descriptor() -> meshlode_metatensor.MeshPotential:
    return meshlode_metatensor.MeshPotential(
        atomic_smearing=1.0,
    )


# Test correct filling of zero and empty blocks when setting global atomic numbers
def test_all_atomic_numbers():
    all_atomic_numbers = [9, 1, 8]
    descriptor = meshlode_metatensor(
        atomic_smearing=1, all_atomic_numbers=all_atomic_numbers
    )
    values = descriptor.compute(toy_system_single_frame())

    for n in all_atomic_numbers:
        assert len(values.block({"center_type": 9, "neighbor_type": n}).values) == 0

    for n in [1, 8]:
        assert torch.equal(
            values.block({"center_type": n, "neighbor_type": 9}).values,
            torch.tensor([[0], [0]]),
        )


# Make sure that the calculators are computing the features without raising errors,
# and returns the correct output format (TensorMap)
def check_operation(calculator):
    descriptor = calculator.compute(toy_system_single_frame())
    assert isinstance(descriptor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert descriptor._type().name() == "TensorMap"


# Run the above test as a normal python script
def test_operation_as_python():
    check_operation(descriptor())


# Similar to the above, but also testing that the code can be compiled as a torch script
def test_operation_as_torch_script():
    scripted = torch.jit.script(descriptor())
    check_operation(scripted)


# Define a more complex toy system consisting of multiple frames, mixing three species.
def toy_system_2() -> List[System]:
    # First few frames containing Nitrogen
    L = 2.0
    frames = []
    frames.append(
        System(
            species=torch.tensor([7]),
            positions=torch.zeros((1, 3)),
            cell=L * 2 * torch.eye(3),
        )
    )
    frames.append(
        System(
            species=torch.tensor([7, 7]),
            positions=torch.zeros((2, 3)),
            cell=L * 2 * torch.eye(3),
        )
    )
    frames.append(
        System(
            species=torch.tensor([7, 7, 7]),
            positions=torch.zeros((3, 3)),
            cell=L * 2 * torch.eye(3),
        )
    )

    # One more frame containing Na and Cl
    positions = torch.tensor([[0, 0, 0], [1.0, 0, 0]])
    cell = torch.tensor([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]])
    frames.append(
        System(species=torch.tensor([11, 17]), positions=positions, cell=cell)
    )

    return frames


class TestMultiFrameToySystem:
    # Compute TensorMap containing features for various hyperparameters, including more
    # extreme values.
    tensormaps_list = []
    frames = toy_system_2()
    for atomic_smearing in [0.01, 0.3, 3.7]:
        for mesh_spacing in [15.3, 0.19]:
            for interpolation_order in [1, 2, 3, 4, 5]:
                MP = meshlode_metatensor.MeshPotential(
                    atomic_smearing=atomic_smearing,
                    mesh_spacing=mesh_spacing,
                    interpolation_order=interpolation_order,
                    subtract_self=False,
                )
                tensormaps_list.append(MP.compute(frames))

    @pytest.mark.parametrize("features", tensormaps_list)
    def test_tensormap_labels(self, features):
        # Test that the keys of the TensorMap for the toy system are correct
        label_values = torch.tensor(
            [
                [7, 7],
                [7, 11],
                [7, 17],
                [11, 7],
                [11, 11],
                [11, 17],
                [17, 7],
                [17, 11],
                [17, 17],
            ]
        )
        label_names = ["center_type", "neighbor_type"]
        labels_ref = metatensor_torch.Labels(names=label_names, values=label_values)

        assert labels_ref == features.keys

    @pytest.mark.parametrize("features", tensormaps_list)
    def test_zero_blocks(self, features):
        # Since the first 3 frames contain Nitrogen only, while the last frame
        # only contains Na and Cl, the features should be zero
        for i in [11, 17]:
            # For structures in which Nitrogen is present, there will be no Na or Cl
            # neighbors. There are six such center atoms in total.
            block = features.block({"center_type": 7, "neighbor_type": i})
            assert torch.equal(block.values, torch.zeros((6, 1)))

            # For structures in which Na or Cl are present, there will be no Nitrogen
            # neighbors.
            block = features.block({"center_type": i, "neighbor_type": 7})
            assert torch.equal(block.values, torch.zeros((1, 1)))

    @pytest.mark.parametrize("features", tensormaps_list)
    def test_nitrogen_blocks(self, features):
        # For this toy data set:
        # - the first frame contains a single atom at the origin
        # - the second frame contains two atoms at the origin
        # - the third frame contains three atoms at the origin
        # Thus, the features should almost be identical, up to a global factor
        # that is the number of atoms (that are exactly on the same position).
        block = features.block({"center_type": 7, "neighbor_type": 7})
        values = block.values[:, 0]  # flatten to 1d
        values_ref = torch.tensor([1.0, 2, 2, 3, 3, 3])

        # We use a slightly higher relative tolerance due to numerical errors
        torch.testing.assert_close(values / values[0], values_ref, rtol=1e-6, atol=0.0)

    @pytest.mark.parametrize("features", tensormaps_list)
    def test_nacl_blocks(self, features):
        # In the NaCl structure, swapping the positions of all Na and Cl atoms leads to
        # an equivalent structure (up to global translation). This leads to symmetry
        # in the features: the Na-density around Cl is the same as the Cl-density around
        # Na and so on.
        block_nana = features.block({"center_type": 11, "neighbor_type": 11})
        block_nacl = features.block({"center_type": 11, "neighbor_type": 17})
        block_clna = features.block({"center_type": 17, "neighbor_type": 11})
        block_clcl = features.block({"center_type": 17, "neighbor_type": 17})
        torch.testing.assert_close(
            block_nacl.values, block_clna.values, rtol=1e-15, atol=0.0
        )
        torch.testing.assert_close(
            block_nana.values, block_clcl.values, rtol=1e-15, atol=0.0
        )
