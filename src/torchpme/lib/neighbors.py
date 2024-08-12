from typing import Optional

import torch


def all_neighbor_indices(
    num_atoms: int,
    dtype: torch.dtype = torch.int64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Computes all neighbor indices between a given number of atoms, excluding self pairs.

    :param num_atoms: number of atoms for which to compute the neighbor indices.
    :param dtype: data type of the returned tensor.
    :param device: The device on which the tensor will be allocated.
    :returns: tensor of shape ``(2, num_atoms * (num_atoms - 1))`` containing all pairs
        excluding self pairs.

    Example
    -------
    >>> neighbor_indices = all_neighbor_indices(num_atoms=3)
    >>> print(neighbor_indices)
    tensor([[1, 2, 0, 2, 0, 1],
            [0, 0, 1, 1, 2, 2]])

    """
    indices = torch.arange(num_atoms, dtype=dtype, device=device).repeat(num_atoms, 1)

    atom_is = indices.flatten()
    atom_js = indices.T.flatten()

    # Filter out the self pairs
    mask = atom_is != atom_js

    return torch.vstack((atom_is[mask], atom_js[mask]))


def distances(
    positions: torch.Tensor,
    neighbor_indices: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
    neighbor_shifts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the pairwise distances based on positions and neighbor indices.

    :param positions: Tensor of shape ``(num_atoms, 3)`` containing the positions of
        each atom.
    :param neighbor_indices: Tensor of shape ``(2, num_pairs)`` containing pairs of atom
        indices.
    :param cell: Optional tensor of shape ``(3, 3)`` representing the periodic boundary
        conditions (PBC) cell vectors.
    :param neighbor_shifts: Optional tensor of shape ``(num_pairs, 3)`` representing the
        shift vectors for each neighbor pair under PBC.
    :returns: Tensor of shape ``(num_pairs,)`` containing the distances between each
        pair of neighbors.

    :raises ValueError: If `cell` is provided without `neighbor_shifts` or vice versa.

    Example
    -------
    >>> import torch
    >>> positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> neighbor_indices = torch.tensor([[0, 0, 1], [1, 2, 2]])
    >>> dists = distances(positions, neighbor_indices)
    >>> print(dists)
    tensor([1.0000, 1.0000, 1.4142])

    If periodic boundary conditions are applied:

    >>> cell = torch.eye(3)  # Identity matrix for cell vectors
    >>> neighbor_shifts = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    >>> dists = distances(positions, neighbor_indices, cell, neighbor_shifts)
    >>> print(dists)
    tensor([1.0000, 1.4142, 1.4142])
    """
    atom_is = neighbor_indices[0]
    atom_js = neighbor_indices[1]

    pos_is = positions[atom_is]
    pos_js = positions[atom_js]

    distance_vectors = pos_js - pos_is

    if cell is not None and neighbor_shifts is not None:
        shifts = neighbor_shifts.type(cell.dtype)
        distance_vectors += shifts @ cell
    elif cell is not None and neighbor_shifts is None:
        raise ValueError("Provided `cell` but no `neighbor_shifts`.")
    elif cell is None and neighbor_shifts is not None:
        raise ValueError("Provided `neighbor_shifts` but no `cell`.")

    return torch.linalg.norm(distance_vectors, dim=1)
