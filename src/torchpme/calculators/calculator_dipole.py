from typing import Optional

import torch
from torch import profiler

from .._utils import _validate_parameters
from ..lib import generate_kvectors_for_ewald
from ..potentials import PotentialDipole


class CalculatorDipole(torch.nn.Module):
    """
    Base calculator for interacting dipoles in the torch interface.

    :param potential: a :class:`PotentialDipole` class object containing the functions
        that are necessary to compute the various components of the potential, as
        well as the parameters that determine the behavior of the potential itself.
    :param full_neighbor_list: parameter indicating whether the neighbor information
        will come from a full (True) or half (False, default) neighbor list.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
    :param lr_wavelength: the wavelength of the long-range part of the potential.
    """

    def __init__(
        self,
        potential: PotentialDipole,
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
        lr_wavelength: Optional[float] = None,
    ):
        super().__init__()

        if not isinstance(potential, PotentialDipole):
            raise TypeError(
                f"Potential must be an instance of PotentialDipole, got {type(potential)}"
            )

        self.potential = potential
        self.lr_wavelength = lr_wavelength

        assert (
            self.lr_wavelength is not None
            and self.potential.smearing is not None
            or (self.lr_wavelength is None and self.potential.smearing is None)
        ), "Either both `lr_wavelength` and `smearing` must be set or both must be None"

        self.full_neighbor_list = full_neighbor_list

        self.prefactor = prefactor

    def _compute_rspace(
        self,
        dipoles: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_vectors: torch.Tensor,
    ) -> torch.Tensor:
        # Compute the pair potential terms V(r_ij) for each pair of atoms (i,j)
        # contained in the neighbor list
        with profiler.record_function("compute bare potential"):
            if self.potential.smearing is None:
                potentials_bare = self.potential.from_dist(neighbor_vectors)
            else:
                potentials_bare = self.potential.sr_from_dist(neighbor_vectors)

        # Multiply the bare potential terms V(r_ij) with the corresponding dipoles
        # of ``atom j'' to obtain q_j*V(r_ij). Since each atom j can be a neighbor of
        # multiple atom i's, we need to access those from neighbor_indices
        atom_is = neighbor_indices[:, 0]
        atom_js = neighbor_indices[:, 1]
        with profiler.record_function("compute real potential"):
            contributions_is = torch.bmm(
                potentials_bare, dipoles[atom_js].unsqueeze(-1)
            ).squeeze(-1)

        # For each atom i, add up all contributions of the form q_j*V(r_ij) for j
        # ranging over all of its neighbors.
        with profiler.record_function("assign potential"):
            potential = torch.zeros_like(dipoles)
            potential.index_add_(0, atom_is, contributions_is)
            # If we are using a half neighbor list, we need to add the contributions
            # from the "inverse" pairs (j, i) to the atoms i
            if not self.full_neighbor_list:
                contributions_js = torch.bmm(
                    potentials_bare, dipoles[atom_is].unsqueeze(-1)
                ).squeeze(-1)
                potential.index_add_(0, atom_js, contributions_js)

        # Compensate for double counting of pairs (i,j) and (j,i)
        return potential / 2

    def _compute_kspace(
        self,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Define k-space cutoff from required real-space resolution
        k_cutoff = 2 * torch.pi / self.lr_wavelength

        # Compute number of times each basis vector of the reciprocal space can be
        # scaled until the cutoff is reached
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_float = k_cutoff * basis_norms / 2 / torch.pi
        ns = torch.ceil(ns_float).long()

        # Generate k-vectors and evaluate
        kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell)
        knorm_sq = torch.sum(kvectors**2, dim=1)
        # We remove the singularity at k=0 by explicitly setting its
        # value to be equal to zero. This mathematically corresponds
        # to the requirement that the net charge of the cell is zero.
        # G = 4 * torch.pi * torch.exp(-0.5 * smearing**2 * knorm_sq) / knorm_sq
        G = self.potential.lr_from_k_sq(knorm_sq)

        # Compute the energy using the explicit method that
        # follows directly from the Poisson summation formula.
        # For this, we precompute trigonometric factors for optimization, which leads
        # to N^2 rather than N^3 scaling.
        trig_args = kvectors @ (positions.T)  # [k, i]
        c = torch.cos(trig_args)  # [k, i]
        s = torch.sin(trig_args)  # [k, i]
        sc = torch.stack([c, s], dim=0)  # [2 "f", k, i]
        mu_k = dipoles @ kvectors.T  # [i, k]
        sc_summed_G = torch.einsum("fki, ik, k->fk", sc, mu_k, G)
        energy = torch.einsum("fk, fki, kc->ic", sc_summed_G, sc, kvectors)
        energy /= torch.abs(cell.det())
        energy -= dipoles * self.potential.self_contribution()
        energy += self.potential.background_correction(
            torch.abs(cell.det())
        ) * dipoles.sum(dim=0)
        return energy / 2

    def forward(
        self,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_vectors: torch.Tensor,
    ):
        r"""
        Compute the potential "energy".

        It is calculated as:

        .. math::

            V_i = \frac{1}{2} \sum_{j} \boldsymbol{\mu_j} \, \mathbf{v}(\mathbf{r_{ij}})

        where :math:`\mathbf{v}(\mathbf{r})` is the pair potential defined by the ``potential``
        parameter, and :math:`\boldsymbol{\mu_j}` are atomic "dipoles".

        If the ``smearing`` of the ``potential`` is not set, the calculator evaluates
        only the real-space part of the potential. Otherwise, provided that the
        calculator implements a ``_compute_kspace`` method, it will also evaluate the
        long-range part using a Fourier-domain method.

        :param dipoles: torch.tensor of shape ``(len(positions), 3)``
            containaing the atomic dipoles.
        :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
            vector of the unit cell
        :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
            coordinates of the ``N`` particles within the supercell.
        :param neighbor_indices: torch.tensor with the ``i,j`` indices of neighbors for
            which the potential should be computed in real space.
        :param neighbor_vectors: torch.tensor with the pair vectors of the neighbors
            for which the potential should be computed in real space.
        """
        # TODO: _validate_parameters to allow also dipoles. Temporarily pass the
        # distance tensor.
        _validate_parameters(
            charges=dipoles,
            cell=cell,
            positions=positions,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_vectors.norm(dim=-1),
            smearing=self.potential.smearing,
        )

        # Compute short-range (SR) part using a real space sum
        potential_sr = self._compute_rspace(
            dipoles=dipoles,
            neighbor_indices=neighbor_indices,
            neighbor_vectors=neighbor_vectors,
        )

        if self.potential.smearing is None:
            return self.prefactor * potential_sr
        # Compute long-range (LR) part using a Fourier / reciprocal space sum
        potential_lr = self._compute_kspace(
            dipoles=dipoles,
            cell=cell,
            positions=positions,
        )

        return self.prefactor * (potential_sr + potential_lr)
