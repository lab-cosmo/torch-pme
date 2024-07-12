from typing import List, Optional, Union

import torch

from .base import CalculatorBaseTorch


class _DirectPotentialImpl:
    def __init__(self, exponent):
        self.exponent = exponent

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        cell: None,
        charges: torch.Tensor,
        neighbor_indices: None,
        neighbor_shifts: None,
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


class DirectPotential(CalculatorBaseTorch, _DirectPotentialImpl):
    r"""Specie-wise long-range potential using a direct summation over all atoms.

    Scaling as :math:`\mathcal{O}(N^2)` with respect to the number of particles
    :math:`N`. As opposed to the Ewald sum, this calculator does NOT take into account
    periodic images, and it will instead be assumed that the provided atoms are in the
    infinitely extended three-dimensional Euclidean space. While slow, this
    implementation used as a reference to test faster algorithms.

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials

    Example
    -------
    We compute the energy of two charges which are sepearated by 2 along the z-axis.

    >>> import torch

    Define simple example structure

    >>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    >>> charges = torch.tensor([1.0, -1.0]).reshape(-1, 1)

    Compute features

    >>> direct = DirectPotential()
    >>> direct.compute(positions=positions, charges=charges)
    tensor([[-0.5000],
            [ 0.5000]])

    Which is the expected potential since :math:`V \propto 1/r` where :math:`r` is the
    distance between the particles.
    """

    def __init__(self, exponent: float = 1.0):
        _DirectPotentialImpl.__init__(self, exponent=exponent)
        CalculatorBaseTorch.__init__(self)

    def compute(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute potential for all provided "systems" stacked inside list.

        The computation is performed on the same ``device`` as ``dtype`` is the input is
        stored on. The ``dtype`` of the output tensors will be the same as the input.

        :param positions: Single or 2D tensor of shape (``len(charges), 3``) containing
            the Cartesian positions of all point charges in the system.
        :param charges: Single 2D tensor or list of 2D tensor of shape (``n_channels,
            len(positions))``. ``n_channels`` is the number of charge channels the
            potential should be calculated for a standard potential ``n_channels=1``. If
            more than one "channel" is provided multiple potentials for the same
            position but different are computed.
        :return: Single or List of torch Tensors containing the potential(s) for all
            positions. Each tensor in the list is of shape ``(len(positions),
            len(charges))``, where If the inputs are only single tensors only a single
            torch tensor with the potentials is returned.
        """

        return self._compute_impl(
            positions=positions,
            cell=None,
            charges=charges,
            neighbor_indices=None,
            neighbor_shifts=None,
        )

    # This function is kept to keep MeshLODE compatible with the broader pytorch
    # infrastructure, which require a "forward" function. We name this function
    # "compute" instead, for compatibility with other COSMO software.
    def forward(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(
            positions=positions,
            charges=charges,
        )
