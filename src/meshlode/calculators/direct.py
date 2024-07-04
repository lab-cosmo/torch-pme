from typing import Union

import torch

from .calculator_base import CalculatorBase


class DirectPotential(CalculatorBase):
    """A specie-wise long-range potential computed using a direct summation over all
    pairs of atoms, scaling as O(N^2) with respect to the number of particles N.
    As opposed to the Ewald sum, this calculator does NOT take into account periodic
    images, and it will instead be assumed that the provided atoms are in the infinitely
    extended three-dimensional Euclidean space.
    While slow, this implementation used as a reference to test faster algorithms.

    :param all_types: Optional global list of all atomic types that should be considered
        for the computation. This option might be useful when running the calculation on
        subset of a whole dataset and it required to keep the shape of the output
        consistent. If this is not set the possible atomic types will be determined when
        calling the :meth:`compute()`.
    """

    name = "DirectPotential"

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        cell: Union[None, torch.Tensor],
        charges: torch.Tensor,
        neighbor_indices: Union[None, torch.Tensor],
        neighbor_shifts: Union[None, torch.Tensor],
    ) -> torch.Tensor:
        # Compute matrix containing the squared distances from the Gram matrix
        # The squared distance and the inner product between two vectors r_i and r_j are
        # related by: d_ij^2 = |r_i - r_j|^2 = r_i^2 + r_j^2 - 2*r_i*r_j
        num_atoms = len(positions)
        diagonal_indices = torch.arange(num_atoms)
        gram_matrix = positions @ positions.T
        squared_norms = gram_matrix[diagonal_indices, diagonal_indices].reshape(-1, 1)
        ones = torch.ones((1, len(positions)), dtype=positions.dtype)
        squared_norms_matrix = torch.matmul(squared_norms, ones)
        distances_sq = squared_norms_matrix + squared_norms_matrix.T - 2 * gram_matrix

        # Add terms to diagonal in order to avoid division by zero
        # Since these components in the target tensor need to be set to zero, we add
        # a huge number such that after taking the inverse (since we evaluate 1/r^p),
        # the components will effectively be set to zero.
        # This is not the most elegant solution, but I am doing this since the more
        # obvious alternative of setting the same components to zero after the division
        # had issues with autograd. I would appreciate any better alternatives.
        distances_sq[diagonal_indices, diagonal_indices] += 1e50

        # Compute potential
        potentials_by_pair = distances_sq.pow(-self.exponent / 2.0)

        return torch.matmul(potentials_by_pair, charges)
