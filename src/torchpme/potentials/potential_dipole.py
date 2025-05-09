from typing import Optional

import torch

from .potential import Potential


class PotentialDipole(torch.nn.Module):
    r"""
    Pair potential energy function between point dipoles.

    The intercation is described as

    .. math::

        V(\vec{r}) = \frac{(\vec{\mu}_i \cdot \vec{\mu}_j)}{r^3} -
            \frac{3  (\vec{\mu}_i \cdot \vec{r})  (\vec{\mu}_j \cdot \vec{r}) }{r^5}

    where :math:`r=|\vec{r}|`.

    :param smearing: float or torch.Tensor containing the parameter often called "sigma"
        in publications, which determines the length-scale at which the short-range and
        long-range parts of the naive :math:`1/r` potential are separated. The smearing
        parameter corresponds to the "width" of a Gaussian smearing of the particle
        density.
    :param exclusion_radius: A length scale that defines a *local environment* within
        which the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param epsilon: Dielectric constant of the medium in which the dipoles are embedded.
    :param exclusion_degree: Controls the sharpness of the transition in the cutoff function
        applied within the ``exclusion_radius``. The cutoff is computed as a raised cosine
        with exponent ``exclusion_degree``
    """

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
        epsilon: float = 0.0,
    ):
        super().__init__()

        self.exclusion_degree = exclusion_degree
        if smearing is not None:
            self.register_buffer(
                "smearing", torch.tensor(smearing, dtype=torch.float64)
            )
        else:
            self.smearing = None
        if exclusion_radius is not None:
            self.register_buffer(
                "exclusion_radius",
                torch.tensor(exclusion_radius, dtype=torch.float64),
            )
        else:
            self.exclusion_radius = None
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float64))

    @torch.jit.export
    def f_cutoff(self, vector: torch.Tensor) -> torch.Tensor:
        r"""
        Default cutoff function defining the *local* region that should be excluded from
        the computation of a long-range model. Defaults to a shifted cosine
        :math:`1 - ((1 - \cos \pi r/r_\mathrm{cut})/2) ^ n`. where :math:`n` is the
        ``exclusion_degree`` parameter.

        :param vector: torch.tensor containing the vectors at which the potential is to
            be evaluated.
        """
        r_mag = torch.norm(vector, dim=1, keepdim=True)
        if self.exclusion_radius is None:
            raise ValueError(
                "Cannot compute cutoff function when `exclusion_radius` is not set"
            )

        return torch.where(
            r_mag < self.exclusion_radius,
            1
            - ((1 - torch.cos(torch.pi * (r_mag / self.exclusion_radius))) * 0.5)
            ** self.exclusion_degree,
            0.0,
        )

    def from_dist(self, vector: torch.Tensor) -> torch.Tensor:
        r"""
        Full dipolar potential as a function of :math:`\mathbf{r}`.

        :param vector: torch.tensor containing the vectors at which the potential is to
            be evaluated.
        """
        r_mag = torch.norm(vector, dim=1, keepdim=True)
        scalar_potential = 1.0 / (r_mag**3)
        r_outer = torch.bmm(vector.unsqueeze(2), vector.unsqueeze(1))
        return scalar_potential.unsqueeze(-1) * torch.eye(3).to(r_outer).unsqueeze(
            0
        ) - 3.0 * r_outer / (r_mag**5).unsqueeze(-1)

    @torch.jit.export
    def sr_from_dist(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Short-range part of the pair potential in real space.

        :param dist: torch.tensor containing the distance vectors at which the potential
            is to be evaluated.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute range-separated potential when `smearing` is not specified."
            )
        if self.exclusion_radius is None:
            return self.from_dist(vector) - self.lr_from_dist(vector)
        return -self.lr_from_dist(vector) * self.f_cutoff(vector).unsqueeze(-1)

    @torch.jit.export
    def lr_from_dist(self, vector: torch.Tensor) -> torch.Tensor:
        r"""
        Long-range of the range-separated dipolar potential.

        Used to subtract out the interior contributions after computing the long-range
        part in reciprocal (Fourier) space.

        :param vector: torch.tensor containing the vectors at which the potential is to
            be evaluated.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )
        alpha = 1 / (2 * self.smearing**2)
        r_mag = torch.norm(vector, dim=1, keepdim=True)
        r_outer = torch.bmm(vector.unsqueeze(2), vector.unsqueeze(1))
        B1 = torch.erfc(torch.sqrt(alpha) * r_mag) / r_mag**3
        B2 = 2 * torch.sqrt(alpha / torch.pi) * torch.exp(-alpha * r_mag**2) / r_mag**2
        B = 1.0 / (r_mag**3) - B1 - B2
        C1 = 3.0 * torch.erfc(torch.sqrt(alpha) * r_mag) / r_mag**5
        C2 = (
            2
            * torch.sqrt(alpha / torch.pi)
            * (2 * alpha + 3 / r_mag**2)
            * torch.exp(-alpha * r_mag**2)
            / r_mag**2
        )
        C = 3.0 / (r_mag**5) - C1 - C2
        return B.unsqueeze(-1) * torch.eye(3).to(r_outer).unsqueeze(
            0
        ) - r_outer * C.unsqueeze(-1)

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        r"""
        Fourier transform of the long-range part of the potential.

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
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )
        alpha = 1 / (2 * self.smearing**2)
        return 4 * torch.pi / 3 * torch.sqrt((alpha / torch.pi) ** 3)

    def background_correction(self, volume) -> torch.Tensor:
        if self.epsilon == 0.0:
            return self.epsilon
        return 4 * torch.pi / (2 * self.epsilon + 1) / volume

    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
