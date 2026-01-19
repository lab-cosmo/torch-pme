from typing import Optional

import torch

from .potential import Potential


def _pbc_correction(
    periodic: Optional[torch.Tensor],
    positions: torch.Tensor,
    cell: torch.Tensor,
    charges: torch.Tensor,
) -> torch.Tensor:
    # Define this helper function as this function is used in multiple potentials

    # "2D periodicity" correction for 1/r potential
    if periodic is None:
        periodic = torch.tensor([True, True, True], device=cell.device)
    n_periodic = torch.sum(periodic)
    is_2d = n_periodic == 2
    axis = torch.argmax(
        torch.where(
            is_2d.unsqueeze(-1),
            (~periodic).to(torch.int64),
            torch.zeros_like(periodic, dtype=torch.int64),
        ),
        dim=-1,
    )
    E_slab = torch.zeros_like(charges)
    z_i = torch.gather(positions, 1, axis.expand(positions.shape[0]).unsqueeze(-1))
    basis_len = torch.gather(torch.linalg.norm(cell, dim=-1), 0, axis)
    V = torch.abs(torch.linalg.det(cell))
    charge_tot = torch.sum(charges, dim=0)
    M_axis = torch.sum(charges * z_i, dim=0)
    M_axis_sq = torch.sum(charges * z_i**2, dim=0)
    E_slab_2d = (4.0 * torch.pi / V) * (
        z_i * M_axis
        - 0.5 * (M_axis_sq + charge_tot * z_i**2)
        - charge_tot / 12.0 * basis_len**2
    )

    return torch.where(is_2d.unsqueeze(-1), E_slab_2d, E_slab)


class CoulombPotential(Potential):
    """
    Smoothed electrostatic Coulomb potential :math:`1/r`.

    Here :math:`r` is the inter-particle distance

    It can be used to compute:

    1. the full :math:`1/r` potential
    2. its short-range (SR) and long-range (LR) parts, the split being determined by a
       length-scale parameter (called "Inverse" in the code)
    3. the Fourier transform of the LR part

    :param smearing: float or torch.Tensor containing the parameter often called "sigma"
        in publications, which determines the length-scale at which the short-range and
        long-range parts of the naive :math:`1/r` potential are separated. The smearing
        parameter corresponds to the "width" of a Gaussian smearing of the particle
        density.
    :param exclusion_radius: A length scale that defines a *local environment* within
        which the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param exclusion_degree: Controls the sharpness of the transition in the cutoff function
        applied within the ``exclusion_radius``. The cutoff is computed as a raised cosine
        with exponent ``exclusion_degree``
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
    """

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
        prefactor: float = 1.0,
    ):
        super().__init__(smearing, exclusion_radius, exclusion_degree, prefactor)

    def from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full :math:`1/r` potential as a function of :math:`r`.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        :param pair_mask: Optional torch.tensor containing a mask to be applied to the
            result.
        """
        result = 1.0 / dist.clamp(min=1e-15)

        if pair_mask is not None:
            result = result * pair_mask  # elementwise multiply, keeps shape fixed

        return self.prefactor * result

    def lr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Long range of the range-separated :math:`1/r` potential.

        Used to subtract out the interior contributions after computing the LR part in
        reciprocal (Fourier) space.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        :param pair_mask: Optional torch.tensor containing a mask to be applied to the
            result.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )
        result = torch.erf(dist / self.smearing / 2.0**0.5) / dist.clamp(min=1e-12)
        if pair_mask is not None:
            result = result * pair_mask  # elementwise multiply, keeps shape fixed

        return self.prefactor * result

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        r"""
        Fourier transform of the LR part potential in terms of :math:`\mathbf{k^2}`.

        :param k_sq: torch.tensor containing the squared lengths (2-norms) of the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range kernel without specifying `smearing`."
            )

        # avoid NaNs in backward, see
        # https://github.com/jax-ml/jax/issues/1052
        # https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
        masked = torch.where(k_sq == 0, 1.0, k_sq)
        return self.prefactor * torch.where(
            k_sq == 0,
            0.0,
            4 * torch.pi * torch.exp(-0.5 * self.smearing**2 * masked) / masked,
        )

    def self_contribution(self) -> torch.Tensor:
        # self-correction for 1/r potential
        if self.smearing is None:
            raise ValueError(
                "Cannot compute self contribution without specifying `smearing`."
            )
        return self.prefactor * (2 / torch.pi) ** 0.5 / self.smearing

    def background_correction(self) -> torch.Tensor:
        # "charge neutrality" correction for 1/r potential
        if self.smearing is None:
            raise ValueError(
                "Cannot compute background correction without specifying `smearing`."
            )
        return self.prefactor * torch.pi * self.smearing**2

    def pbc_correction(
        self,
        periodic: Optional[torch.Tensor],
        positions: torch.Tensor,
        cell: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        return self.prefactor * _pbc_correction(periodic, positions, cell, charges)

    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
    pbc_correction.__doc__ = Potential.pbc_correction.__doc__
