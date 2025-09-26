from typing import Optional

import torch

from .potential import Potential


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
    """

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
    ):
        super().__init__(smearing, exclusion_radius, exclusion_degree)

    def from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Full :math:`1/r` potential as a function of :math:`r`.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """
        return 1.0 / dist

    def lr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Long range of the range-separated :math:`1/r` potential.

        Used to subtract out the interior contributions after computing the LR part in
        reciprocal (Fourier) space.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )

        return torch.erf(dist / self.smearing / 2.0**0.5) / dist

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
        return torch.where(
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
        return (2 / torch.pi) ** 0.5 / self.smearing

    def background_correction(self) -> torch.Tensor:
        # "charge neutrality" correction for 1/r potential
        if self.smearing is None:
            raise ValueError(
                "Cannot compute background correction without specifying `smearing`."
            )
        return torch.pi * self.smearing**2

    @staticmethod
    def pbc_correction(
        periodic: Optional[torch.Tensor],
        positions: torch.Tensor,
        cell: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        # "2D periodicity" correction for 1/r potential
        if periodic is None:
            periodic = torch.tensor([True, True, True], device=cell.device)

        n_periodic = torch.sum(periodic).item()
        if n_periodic == 3:
            periodicity = 3
            nonperiodic_axis = None
        elif n_periodic == 2:
            periodicity = 2
            nonperiodic_axis = torch.where(~periodic)[0]
            max_distance = torch.max(positions[:, nonperiodic_axis]) - torch.min(
                positions[:, nonperiodic_axis]
            )
            cell_size = torch.linalg.norm(cell[nonperiodic_axis])
            if max_distance > cell_size / 3:
                raise ValueError(
                    f"Maximum distance along non-periodic axis ({max_distance}) "
                    f"exceeds one third of cell size ({cell_size})."
                )
        else:
            raise ValueError(
                "K-space summation is not implemented for 1D or non-periodic systems."
            )

        if periodicity == 2:
            charge_tot = torch.sum(charges, dim=0)
            axis = nonperiodic_axis
            z_i = positions[:, axis].view(-1, 1)
            basis_len = torch.linalg.norm(cell[axis])
            M_axis = torch.sum(charges * z_i, dim=0)
            M_axis_sq = torch.sum(charges * z_i**2, dim=0)
            V = torch.abs(torch.linalg.det(cell))
            E_slab = (4.0 * torch.pi / V) * (
                z_i * M_axis
                - 0.5 * (M_axis_sq + charge_tot * z_i**2)
                - charge_tot / 12.0 * basis_len**2
            )
        else:
            E_slab = torch.zeros_like(charges)

        return E_slab

    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
    pbc_correction.__doc__ = Potential.pbc_correction.__doc__
