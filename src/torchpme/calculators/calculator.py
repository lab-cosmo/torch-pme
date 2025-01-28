from typing import Optional, Union

import torch
from torch import profiler

from .._utils import _validate_parameters
from ..potentials import Potential


class Calculator(torch.nn.Module):
    """
    Base calculator for the torch interface. Based on a
    :class:`Potential` class, it computes the value of a potential
    by either directly summing over neighbor atoms, or by combining
    a local part computed in real space, and a long-range part computed
    in the Fourier domain. The class can be used directly to evaluate
    the real-space part of the potential, or subclassed providing
    a strategy to evalate the long-range contribution in k-space
    (see e.g. :class:`PMECalculator` or :class:`EwaldCalculator`).
    NB: typically a subclass should only provide an implementation of
    :func:`Calculator._compute_kspace`.

    :param potential: a :class:`Potential` class object containing the functions
        that are necessary to compute the various components of the potential, as
        well as the parameters that determine the behavior of the potential itself.
    :param full_neighbor_list: parameter indicating whether the neighbor information
        will come from a full (True) or half (False, default) neighbor list.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
    :param dtype: type used for the internal buffers and parameters
    :param device: device used for the internal buffers and parameters
    """

    def __init__(
        self,
        potential: Potential,
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Union[None, str, torch.device] = None,
    ):
        super().__init__()

        if not isinstance(potential, Potential):
            raise TypeError(
                f"Potential must be an instance of Potential, got {type(potential)}"
            )

        self.device = torch.get_default_device() if device is None else device
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

        if self.dtype != potential.dtype:
            raise TypeError(
                f"dtype of `potential` ({potential.dtype}) must be same as of "
                f"`calculator` ({self.dtype})"
            )

        if self.device != potential.device:
            raise ValueError(
                f"device of `potential` ({potential.device}) must be same as of "
                f"`calculator` ({self.device})"
            )

        self.potential = potential
        self.full_neighbor_list = full_neighbor_list

        self.prefactor = prefactor

    def _compute_rspace(
        self,
        charges: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the "real space" part of the potential. Depending on the
        ``smearing`` in the potential class, it will either evaluate the
        full potential, or only the short-range, "local" part.

        :param charges: 2D tensor or list of 2D tensor of shape
            ``(n_channels,len(positions))``.
            ``n_channels`` is the number of charge channels the
            potential should be calculated for a standard potential ``n_channels=1``.
            If more than one "channel" is provided multiple potentials for the same
            position are computed depending on the charges and the potentials.
        :param neighbor_indices: Single or list of 2D tensors of shape ``(n, 2)``, where
            ``n`` is the number of neighbors. The two columns correspond to the indices
            of a **half neighbor list** for the two atoms which are considered neighbors
            (e.g. within a cutoff distance) if ``full_neighbor_list=False`` (default).
            Otherwise, a full neighbor list is expected.
        :param neighbor_distances: single or list of 1D tensors containing the distance
            between the ``n`` pairs corresponding to a **half (or full) neighbor list**
            (see ``neighbor_indices``).
        :return: Torch tensor containing the potential(s) for all
            positions. Each tensor in the list is of shape ``(len(positions),
            len(charges))``, where If the inputs are only single tensors only a single
            torch tensor with the potentials is returned.
        """
        # Compute the pair potential terms V(r_ij) for each pair of atoms (i,j)
        # contained in the neighbor list
        with profiler.record_function("compute bare potential"):
            if self.potential.smearing is None:
                potentials_bare = self.potential.from_dist(neighbor_distances)
            else:
                potentials_bare = self.potential.sr_from_dist(neighbor_distances)

        # Multiply the bare potential terms V(r_ij) with the corresponding charges
        # of ``atom j'' to obtain q_j*V(r_ij). Since each atom j can be a neighbor of
        # multiple atom i's, we need to access those from neighbor_indices
        atom_is = neighbor_indices[:, 0]
        atom_js = neighbor_indices[:, 1]
        with profiler.record_function("compute real potential"):
            contributions_is = charges[atom_js] * potentials_bare.unsqueeze(-1)

        # For each atom i, add up all contributions of the form q_j*V(r_ij) for j
        # ranging over all of its neighbors.
        with profiler.record_function("assign potential"):
            potential = torch.zeros_like(charges)
            potential.index_add_(0, atom_is, contributions_is)
            # If we are using a half neighbor list, we need to add the contributions
            # from the "inverse" pairs (j, i) to the atoms i
            if not self.full_neighbor_list:
                contributions_js = charges[atom_is] * potentials_bare.unsqueeze(-1)
                potential.index_add_(0, atom_js, contributions_js)

        # Compensate for double counting of pairs (i,j) and (j,i)
        return potential / 2

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the Fourier-domain contribution to the potential, typically
        corresponding to a long-range, slowly decaying type of interaction.
        """
        raise NotImplementedError(
            f"`compute_kspace` not implemented for {self.__class__.__name__}"
        )

    def forward(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ):
        r"""
        Compute the potential "energy".

        It is calculated as

        .. math::

            V_i = \frac{1}{2} \sum_{j} q_j\,v(r_{ij})

        where :math:`v(r)` is the pair potential defined by the ``potential`` parameter
        and :math:`q_j` are atomic "charges" (corresponding to the electrostatic charge
        when using a Coulomb potential).

        If the ``smearing`` of the ``potential`` is not set, the calculator evaluates
        only the real-space part of the potential. Otherwise, provided that the
        calculator implements a ``_compute_kspace`` method, it will also evaluate the
        long-range part using a Fourier-domain method.

        :param charges: torch.Tensor, atomic (pseudo-)charges
        :param cell: torch.Tensor, periodic supercell for the system
        :param positions: torch.Tensor, Cartesian coordinates of the particles within
            the supercell.
        :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
            which the potential should be computed in real space.
        :param neighbor_distances: torch.Tensor with the pair distances of the neighbors
            for which the potential should be computed in real space.
        """
        _validate_parameters(
            charges=charges,
            cell=cell,
            positions=positions,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            smearing=self.potential.smearing,
            dtype=self.dtype,
            device=self.device,
        )

        # Compute short-range (SR) part using a real space sum
        potential_sr = self._compute_rspace(
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )

        if self.potential.smearing is None:
            return self.prefactor * potential_sr
        # Compute long-range (LR) part using a Fourier / reciprocal space sum
        potential_lr = self._compute_kspace(
            charges=charges,
            cell=cell,
            positions=positions,
        )

        return self.prefactor * (potential_sr + potential_lr)
