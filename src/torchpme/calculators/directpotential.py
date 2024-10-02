import torch

from ..lib import InversePowerLawPotential
from .base import BaseCalculator


class DirectPotential(BaseCalculator):
    r"""
    Potential using a direct summation.

    Scaling as :math:`\mathcal{O}(N^2)` with respect to the number of particles
    :math:`N`. As opposed to the Ewald sum, this calculator does NOT take into account
    periodic images, and it will instead be assumed that the provided atoms are in the
    infinitely extended three-dimensional Euclidean space. While slow, this
    implementation used as a reference to test faster algorithms.

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.

    Example
    -------
    We compute the energy of two charges which are sepearated by 2 along the z-axis.

    >>> import torch
    >>> from vesin.torch import NeighborList

    Define simple example structure

    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)

    Compute the neighbor indices (``"i"``, ``"j"``) and the neighbor distances ("``d``")
    using the ``vesin`` package. Refer to the `documentation <https://luthaf.fr/vesin>`_
    for details on the API. We define a dummy cell

    >>> cell = torch.eye(3, dtype=torch.float64)

    The ``cell`` and ``positions`` argument **are ignored** in the actual calculations
    but is required for a consistent API with the other calculators.

    >>> nl = NeighborList(cutoff=3.0, full_list=False)
    >>> i, j, neighbor_distances = nl.compute(
    ...     points=positions, box=cell, periodic=False, quantities="ijd"
    ... )
    >>> neighbor_indices = torch.stack([i, j], dim=1)

    Finally, we initlize the potential class and ``compute`` the potential for the
    system.

    >>> direct = DirectPotential()
    >>> direct.forward(
    ...     positions=positions,
    ...     charges=charges,
    ...     cell=cell,
    ...     neighbor_indices=neighbor_indices,
    ...     neighbor_distances=neighbor_distances,
    ... )
    tensor([[-0.2500],
            [ 0.2500]], dtype=torch.float64)

    Which is the expected potential since :math:`V \propto 1/r` where :math:`r` is the
    distance between the particles.

    """

    def __init__(self, exponent: float = 1.0, full_neighbor_list: bool = False):
        # Use a dummy value for the smearing. We don't use the methods for smeared
        # potentials in a direct potential.
        super().__init__(
            potential=InversePowerLawPotential(exponent=exponent, smearing=0.0),
            full_neighbor_list=full_neighbor_list,
        )

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        return self._compute_sr(
            is_periodic=False,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            subtract_interior=False,  # ignored since `is_periodic=False`
        )
