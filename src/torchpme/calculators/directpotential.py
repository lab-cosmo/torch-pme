from typing import List, Optional, Union

import torch

from ..lib import InversePowerLawPotential, all_neighbor_indices, distances
from .base import CalculatorBaseTorch


class _DirectPotentialImpl:
    def __init__(self, exponent):
        self.exponent = exponent
        self.potential = InversePowerLawPotential(exponent=exponent)

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: Optional[torch.Tensor],
        neighbor_indices: Optional[torch.Tensor],
        neighbor_shifts: Optional[torch.Tensor],
    ) -> torch.Tensor:

        if neighbor_indices is None:
            neighbor_indices_tensor = all_neighbor_indices(
                len(charges), device=charges.device
            )
        else:
            neighbor_indices_tensor = neighbor_indices

        dists = distances(
            positions=positions,
            cell=cell,
            neighbor_indices=neighbor_indices_tensor,
            neighbor_shifts=neighbor_shifts,
        )

        potentials_bare = self.potential.potential_from_dist(dists)

        atom_is = neighbor_indices_tensor[0]
        atom_js = neighbor_indices_tensor[1]

        contributions = charges[atom_js] * potentials_bare.unsqueeze(-1)

        potential = torch.zeros_like(charges)
        potential.index_add_(0, atom_is, contributions)

        return potential


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
    >>> charges = torch.tensor([1.0, -1.0]).unsqueeze(1)

    Compute features

    >>> direct = DirectPotential()
    >>> direct.compute(positions=positions, charges=charges)
    tensor([[-0.5000],
            [ 0.5000]])

    Which is the expected potential since :math:`V \propto 1/r` where :math:`r` is the
    distance between the particles.

    Note that you can optionally pass ``neighbor_indices`` to the system to restrict the
    computation to certain pairs. See example of :py:class:`torchpme.PMEPotential` for
    details on the process.
    """

    def __init__(self, exponent: float = 1.0):
        _DirectPotentialImpl.__init__(self, exponent=exponent)
        CalculatorBaseTorch.__init__(self)

    def compute(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]] = None,
        neighbor_indices: Union[
            List[Optional[torch.Tensor]], Optional[torch.Tensor]
        ] = None,
        neighbor_shifts: Union[
            List[Optional[torch.Tensor]], Optional[torch.Tensor]
        ] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute potential for all provided "systems" stacked inside list.

        If the optional parameter  ``neighbor_indices`` is provided only those indices
        are taken into account for the compuation. Otherwise all particles are
        considered for computing the potential. If ``cell`` and `neighbor_shifts` are
        given, compuation is performed taking the periodicity of the system into
        account.

        .. warning ::

            When passing the ``neighbor_shifts`` parameter withput explicit
            ``neighbor_indices``, the shape of the ``neighbor_shifts`` must have a shape
            of ``(num_atoms * (num_atoms - 1), 3)``. Also the order of all pairs must
            match the of :py:func:`torchpme.lib.neighbors.all_neighbor_indices`!

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
        :param neighbor_indices: Optional single or list of 2D tensors of shape ``(2,
            n)``, where ``n`` is the number of atoms. The two rows correspond to the
            indices of a **full neighbor list** for the two atoms which are considered
            neighbors (e.g. within a cutoff distance).
        :param neighbor_shifts: Optional single or list of 2D tensors of shape (3, n),
             where n is the number of atoms. The 3 rows correspond to the shift indices
             for periodic images of a **full neighbor list**.
        :return: Single or List of torch Tensors containing the potential(s) for all
            positions. Each tensor in the list is of shape ``(len(positions),
            len(charges))``, where If the inputs are only single tensors only a single
            torch tensor with the potentials is returned.
        """

        return self._compute_impl(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

    # This function is kept to keep torch-pme compatible with the broader pytorch
    # infrastructure, which require a "forward" function. We name this function
    # "compute" instead, for compatibility with other COSMO software.
    def forward(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]] = None,
        neighbor_indices: Union[
            List[Optional[torch.Tensor]], Optional[torch.Tensor]
        ] = None,
        neighbor_shifts: Union[
            List[Optional[torch.Tensor]], Optional[torch.Tensor]
        ] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )
