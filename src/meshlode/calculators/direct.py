from .calculator_base import CalculatorBase

import torch


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
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the "electrostatic" potential at the position of all atoms in a
        structure.
        This solver does not use periodic boundaries, and thus also does not take into
        account potential periodic images.

        :param positions: torch.tensor of shape (n_atoms, 3). Contains the Cartesian
            coordinates of the atoms. The implementation also works if the positions
            are not contained within the unit cell.
        :param charges: torch.tensor of shape `(n_atoms, n_channels)`. In the simplest
            case, this would be a tensor of shape (n_atoms, 1) where charges[i,0] is the
            charge of atom i. More generally, the potential for the same atom positions
            is computed for n_channels independent meshes, and one can specify the
            "charge" of each atom on each of the meshes independently. For standard LODE
            that treats all (atomic) types separately, one example could be: If n_atoms
            = 4 and the types are [Na, Cl, Cl, Na], one could set n_channels=2 and use
            the one-hot encoding charges = torch.tensor([[1,0],[0,1],[0,1],[1,0]]) for
            the charges. This would then separately compute the "Na" potential and "Cl"
            potential. Subtracting these from each other, one could recover the more
            standard electrostatic potential in which Na and Cl have charges of +1 and
            -1, respectively.
        :param cell: torch.tensor of shape `(3, 3)`. Describes the unit cell of the
            structure, where cell[i] is the i-th basis vector. While redundant in this
            particular implementation, the parameter is kept to keep the same inputs as
            the other calculators.

        :returns: torch.tensor of shape `(n_atoms, n_channels)` containing the potential
        at the position of each atom for the `n_channels` independent meshes separately.
        """
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
        potentials_by_pair = distances_sq.pow(-self.exponent / 2.)
        potentials = torch.matmul(potentials_by_pair, charges)

        return potentials
