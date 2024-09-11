import pytest
import torch

from torchpme.lib import all_neighbor_indices, distances


def test_all_neighbor_indices_basic():
    num_atoms = 3
    expected_output = torch.tensor([[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]]).T
    result = all_neighbor_indices(num_atoms)
    assert torch.equal(result, expected_output)


def test_all_neighbor_indices_dtype_device():
    # Test with dtype and device
    num_atoms = 2
    result = all_neighbor_indices(
        num_atoms, dtype=torch.float32, device=torch.device("cpu")
    )
    assert result.dtype == torch.float32
    assert result.device == torch.device("cpu")


def test_all_neighbor_indices_empty():
    # Test with 0 atoms
    num_atoms = 0
    result = all_neighbor_indices(num_atoms)

    assert torch.equal(result, torch.empty((0, 2), dtype=torch.int64))


def test_distances_basic():
    # Test basic distance computation
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    neighbor_indices = torch.tensor([[0, 0, 1], [1, 2, 2]]).T
    expected_output = torch.tensor([1.0000, 1.0000, 1.4142], dtype=torch.float32)
    result = distances(positions, neighbor_indices)
    assert torch.allclose(result, expected_output, atol=1e-4)


def test_distances_with_pbc():
    # Test distance computation with periodic boundary conditions
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    neighbor_indices = torch.tensor([[0, 0, 1], [1, 2, 2]]).T
    cell = torch.eye(3)

    neighbor_shifts = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    expected_output = torch.tensor([1.0000, 1.4142, 1.4142], dtype=torch.float32)

    result = distances(positions, neighbor_indices, cell, neighbor_shifts)
    assert torch.allclose(result, expected_output, atol=1e-4)


def test_distances_missing_neighbor_shifts():
    # Test for error when cell is provided without neighbor_shifts
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    neighbor_indices = torch.tensor([[0, 1]])
    cell = torch.eye(3)

    with pytest.raises(ValueError, match="Provided `cell` but no `neighbor_shifts`."):
        distances(positions, neighbor_indices, cell)


def test_distances_missing_cell():
    # Test for error when neighbor_shifts are provided without cell
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    neighbor_indices = torch.tensor([[0, 1]])
    neighbor_shifts = torch.tensor([[0, 0, 0]])

    with pytest.raises(ValueError, match="Provided `neighbor_shifts` but no `cell`."):
        distances(positions, neighbor_indices, neighbor_shifts=neighbor_shifts)
