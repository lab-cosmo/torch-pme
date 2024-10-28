from typing import Optional

import torch

from .potential import Potential


class CoulombPotential(Potential):
    """Smoothed electrostatic Coulomb potential :math:`1/r`.

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

    .. minigallery::
        :add-heading:

        torchpme.CoulombPotential
    """

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")

        # constants used in the forwward
        self.register_buffer(
            "_rsqrt2",
            torch.rsqrt(torch.tensor(2.0, dtype=dtype, device=device)),
        )
        self.register_buffer(
            "_sqrt_2_on_pi",
            torch.sqrt(torch.tensor(2.0 / torch.pi, dtype=dtype, device=device)),
        )

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

        return torch.erf(dist * (self._rsqrt2 / self.smearing)) / dist

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """
        Fourier transform of the LR part potential in terms of :math:`k^2`.

        :param k_sq: torch.tensor containing the squared lengths (2-norms) of the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        """

        # The k=0 term often needs to be set separately since for exponents p<=3
        # dimension, there is a divergence to +infinity. Setting this value manually
        # to zero physically corresponds to the addition of a uniform background charge
        # to make the system charge-neutral. For p>3, on the other hand, the
        # Fourier-transformed LR potential does not diverge as k->0, and one
        # could instead assign the correct limit. This is not implemented for now
        # for consistency reasons.
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range kernel without specifying `smearing`."
            )

        return torch.where(
            k_sq == 0,
            0.0,
            4 * torch.pi * torch.exp(-0.5 * self.smearing**2 * k_sq) / k_sq,
        )

    def self_contribution(self) -> torch.Tensor:
        # self-correction for 1/r potential
        if self.smearing is None:
            raise ValueError(
                "Cannot compute self contribution without specifying `smearing`."
            )
        return self._sqrt_2_on_pi / self.smearing

    def background_correction(self) -> torch.Tensor:
        # "charge neutrality" correction for 1/r potential
        if self.smearing is None:
            raise ValueError(
                "Cannot compute background correction without specifying `smearing`."
            )
        return torch.pi * self.smearing**2

    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
