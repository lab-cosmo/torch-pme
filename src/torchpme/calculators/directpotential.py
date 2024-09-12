from typing import List, Union

import torch

from ..lib import InversePowerLawPotential
from .base import CalculatorBaseTorch


class _DirectPotentialImpl:
    def __init__(self, exponent):
        self.exponent = exponent
        self.potential = InversePowerLawPotential(exponent=exponent)

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        potentials_bare = self.potential.potential_from_dist(neighbor_distances)

        atom_is = neighbor_indices[:, 0]
        atom_js = neighbor_indices[:, 1]

        contributions = charges[atom_js] * potentials_bare.unsqueeze(-1)

        potential = torch.zeros_like(charges)
        potential.index_add_(0, atom_is, contributions)

        return potential


class DirectPotential(CalculatorBaseTorch, _DirectPotentialImpl):
    r"""Potential using a direct summation.

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
    >>> charges = torch.tensor([1.0, -1.0]).unsqueeze(1)

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
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor],
        neighbor_distances: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute potential for all provided "systems".

        The computation is performed on the same ``device`` as ``dtype`` is the input is
        stored on. The ``dtype`` of the output tensors will be the same as the input.

        :param positions: Single or 2D tensor of shape (``len(charges), 3``) containing
            the Cartesian positions of all point charges in the system.
        :param charges: Single 2D tensor or list of 2D tensor of shape (``n_channels,
            len(positions))``. ``n_channels`` is the number of charge channels the
            potential should be calculated for a standard potential ``n_channels=1``. If
            more than one "channel" is provided multiple potentials for the same
            position but different are computed.
        :param cell: single or 2D tensor of shape (3, 3), describing the bounding
            box/unit cell of the system. Each row should be one of the bounding box
            vector; and columns should contain the x, y, and z components of these
            vectors (i.e. the cell should be given in row-major order).
        :param neighbor_indices: Single or list of 2D tensors of shape ``(n, 2)``, where
            ``n`` is the number of neighbors. The two columns correspond to the indices
            of a **half neighbor list** for the two atoms which are considered neighbors
            (e.g. within a cutoff distance).
        :param neighbor_distances: single or list of 1D tensors containing the distance
            between the ``n`` pairs corresponding to a **half neighbor list**.
        :return: Single or List of torch Tensors containing the potential(s) for all
            positions. Each tensor in the list is of shape ``(len(positions),
            len(charges))``, where If the inputs are only single tensors only a single
            torch tensor with the potentials is returned.
        """

        # Create dummy cell to follow method synopsis
        if isinstance(positions, list):
            cell = len(charges) * [
                torch.zeros(3, 3, dtype=positions[0].dtype, device=positions[0].device)
            ]
            return self._compute_impl(
                positions=positions,
                charges=charges,
                cell=cell,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
            )
        else:
            cell = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)
            return self._compute_impl(
                positions=positions,
                charges=charges,
                cell=cell,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
            )

    # This function is kept to keep torch-pme compatible with the broader pytorch
    # infrastructure, which require a "forward" function. We name this function
    # "compute" instead, for compatibility with other COSMO software.
    def forward(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor],
        neighbor_distances: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(
            positions=positions,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )
