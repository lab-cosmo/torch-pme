from typing import Optional

import torch
from scipy.special import exp1
from torch.special import gammainc, gammaln

from .potential import Potential


def gamma(x: torch.Tensor) -> torch.Tensor:
    """
    (Complete) Gamma function.

    pytorch has not implemented the commonly used (complete) Gamma function. We define
    it in a custom way to make autograd work as in
    https://discuss.pytorch.org/t/is-there-a-gamma-function-in-pytorch/17122
    """
    return torch.exp(gammaln(x))


class CustomExp1(torch.autograd.Function):
    """Custom exponential integral function Exp1(x) to have an autograd-compatible version."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_numpy = input.cpu().numpy() if not input.is_cpu else input.numpy()
        return torch.tensor(exp1(input_numpy), device=input.device, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return -grad_output * torch.exp(-input) / input


def torch_exp1(input):
    """Wrapper for the custom exponential integral function."""
    return CustomExp1.apply(input)


def gammaincc_over_powerlaw(exponent: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Function to compute the regularized incomplete gamma function complement for integer exponents."""
    if exponent == 1:
        return torch.exp(-z) / z
    if exponent == 2:
        return torch.sqrt(torch.pi / z) * torch.erfc(torch.sqrt(z))
    if exponent == 3:
        return torch_exp1(z)
    if exponent == 4:
        return 2 * (
            torch.exp(-z) - torch.sqrt(torch.pi * z) * torch.erfc(torch.sqrt(z))
        )
    if exponent == 5:
        return torch.exp(-z) - z * torch_exp1(z)
    if exponent == 6:
        return (
            (2 - 4 * z) * torch.exp(-z)
            + 4 * torch.sqrt(torch.pi * z**3) * torch.erfc(torch.sqrt(z))
        ) / 3
    raise ValueError(f"Unsupported exponent: {exponent}")


class InversePowerLawPotential(Potential):
    """
    Inverse power-law potentials of the form :math:`1/r^p`.

    Here :math:`r` is a distance parameter and :math:`p` an exponent.

    It can be used to compute:

    1. the full :math:`1/r^p` potential
    2. its short-range (SR) and long-range (LR) parts, the split being determined by a
       length-scale parameter (called "smearing" in the code)
    3. the Fourier transform of the LR part

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials
    :param smearing: float or torch.Tensor containing the parameter often called "sigma"
        in publications, which determines the length-scale at which the short-range and
        long-range parts of the naive :math:`1/r^p` potential are separated. For the
        Coulomb potential (:math:`p=1`), this potential can be interpreted as the
        effective potential generated by a Gaussian charge density, in which case this
        smearing parameter corresponds to the "width" of the Gaussian.
    :param: exclusion_radius: float or torch.Tensor containing the length scale
        corresponding to a local environment. See also
        :class:`Potential`.
    :param dtype: type used for the internal buffers and parameters
    :param device: device used for the internal buffers and parameters
    """

    def __init__(
        self,
        exponent: int,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)

        # function call to check the validity of the exponent
        gammaincc_over_powerlaw(exponent, torch.tensor(1.0, dtype=dtype, device=device))
        self.register_buffer(
            "exponent", torch.tensor(exponent, dtype=self.dtype, device=self.device)
        )

    @torch.jit.export
    def from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Full :math:`1/r^p` potential as a function of :math:`r`.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """
        return torch.pow(dist, -self.exponent)

    @torch.jit.export
    def lr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Long range of the range-separated :math:`1/r^p` potential.

        Used to subtract out the interior contributions after computing the LR part in
        reciprocal (Fourier) space.

        For the Coulomb potential, this would return (note that the only change between
        the SR and LR parts is the fact that erfc changes to erf)

        .. code-block:: python

            potential = erf(dist / sqrt(2) / smearing) / dist

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )

        exponent = self.exponent
        smearing = self.smearing

        x = 0.5 * dist**2 / smearing**2
        peff = exponent / 2
        prefac = 1.0 / (2 * smearing**2) ** peff
        return prefac * gammainc(peff, x) / x**peff

    @torch.jit.export
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

        exponent = self.exponent
        smearing = self.smearing

        peff = (3 - exponent) / 2
        prefac = torch.pi**1.5 / gamma(exponent / 2) * (2 * smearing**2) ** peff
        x = 0.5 * smearing**2 * k_sq

        # The k=0 term often needs to be set separately since for exponents p<=3
        # dimension, there is a divergence to +infinity. Setting this value manually
        # to zero physically corresponds to the addition of a uniform background charge
        # to make the system charge-neutral. For p>3, on the other hand, the
        # Fourier-transformed LR potential does not diverge as k->0, and one
        # could instead assign the correct limit. This is not implemented for now
        # for consistency reasons.
        masked = torch.where(x == 0, 1.0, x)  # avoid NaNs in backwards, see Coulomb
        return torch.where(
            k_sq == 0, 0.0, prefac * gammaincc_over_powerlaw(exponent, masked)
        )

    def self_contribution(self) -> torch.Tensor:
        # self-correction for 1/r^p potential
        if self.smearing is None:
            raise ValueError(
                "Cannot compute self contribution without specifying `smearing`."
            )
        phalf = self.exponent / 2
        return 1 / gamma(phalf + 1) / (2 * self.smearing**2) ** phalf

    def background_correction(self) -> torch.Tensor:
        # "charge neutrality" correction for 1/r^p potential
        if self.smearing is None:
            raise ValueError(
                "Cannot compute background correction without specifying `smearing`."
            )
        prefac = torch.pi**1.5 * (2 * self.smearing**2) ** ((3 - self.exponent) / 2)
        prefac /= (3 - self.exponent) * gamma(self.exponent / 2)
        return prefac

    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
