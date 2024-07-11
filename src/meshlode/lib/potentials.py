import math

import torch
from torch.special import gammainc, gammaincc, gammaln


# since pytorch has implemented the incomplete Gamma functions, but not the much more
# commonly used (complete) Gamma function, we define it in a custom way to make autograd
# work as in https://discuss.pytorch.org/t/is-there-a-gamma-function-in-pytorch/17122
def gamma(x: torch.Tensor):
    return torch.exp(gammaln(x))


class InversePowerLawPotential:
    """
    Class to handle inverse power-law potentials of the form 1/r^p, where r is a
    distance parameter and p an exponent.

    It can be used to compute:
    1. the full 1/r^p potential
    2. its short-range (SR) and long-range  (LR) parts, the split being determined by a
       length-scale parameter (called "smearing" in the code)
    3. the Fourier transform of the LR part

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials
    """

    def __init__(self, exponent: float):
        self.exponent = exponent

    def potential_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Full 1/r^p potential as a function of r

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """
        return torch.pow(dist, -self.exponent)

    def potential_from_dist_sq(self, dist_sq: torch.Tensor) -> torch.Tensor:
        """
        Full 1/r^p potential as a function of r^2, which is more useful in some
        implementations

        :param dist_sq: torch.tensor containing the squared distances at which the
            potential is to be evaluated.
        """
        return torch.pow(dist_sq, -self.exponent / 2.0)

    def potential_sr_from_dist(
        self, dist: torch.Tensor, smearing: float
    ) -> torch.Tensor:
        """
        Short-range (SR) part of the range-separated 1/r^p potential as a function of r.
        More explicitly: it corresponds to V_SR(r) in 1/r^p = V_SR(r) + V_LR(r),
        where the location of the split is determined by the smearing parameter.

        For the Coulomb potential, this would return
        potential = erfc(dist / torch.sqrt(2.) / smearing) / dist

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        :param smearing: float containing the parameter often called "sigma" in
            publications, which determines the length-scale at which the short-range and
            long-range parts of the naive 1/r^p potential are separated. For the Coulomb
            potential (p=1), this potential can be interpreted as the effective
            potential generated by a Gaussian charge density, in which case this
            smearing parameter corresponds to the "width" of the Gaussian.
        """
        exponent = torch.tensor(self.exponent, device=dist.device, dtype=dist.dtype)
        x = 0.5 * dist**2 / smearing**2
        peff = exponent / 2
        prefac = 1.0 / (2 * smearing**2) ** peff
        potential = prefac * gammaincc(peff, x) / x**peff

        # potential = erfc(dist / torch.sqrt(torch.tensor(2.)) / smearing) / dist
        return potential

    def potential_lr_from_dist(
        self, dist: torch.Tensor, smearing: float
    ) -> torch.Tensor:
        """
        Long-range (LR) part of the range-separated 1/r^p potential as a function of r.
        Used to subtract out the interior contributions after computing the LR part
        in reciprocal (Fourier) space.

        For the Coulomb potential, this would return (note that the only change between
        the SR and LR parts is the fact that erfc changes to erf)
        potential = erf(dist / sqrt(2) / smearing) / dist

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        :param smearing: float containing the parameter often called "sigma" in
            publications, which determines the length-scale at which the short-range and
            long-range parts of the naive 1/r^p potential are separated. For the Coulomb
            potential (p=1), this potential can be interpreted as the effective
            potential generated by a Gaussian charge density, in which case this
            smearing parameter corresponds to the "width" of the Gaussian.
        """
        exponent = torch.tensor(self.exponent, device=dist.device, dtype=dist.dtype)
        x = 0.5 * dist**2 / smearing**2
        peff = exponent / 2
        prefac = 1.0 / (2 * smearing**2) ** peff
        potential = prefac * gammainc(peff, x) / x**peff
        return potential

    def potential_fourier_from_k_sq(
        self, k_sq: torch.Tensor, smearing: float
    ) -> torch.Tensor:
        """
        Fourier transform of the long-range (LR) part potential parametrized in terms of
        k^2.
        If only the Coulomb potential is needed, the last line can be replaced by
        fourier = 4 * torch.pi * torch.exp(-0.5 * smearing**2 * k_sq) / k_sq

        :param k_sq: torch.tensor containing the squared lengths (2-norms) of the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        :param smearing: float containing the parameter often called "sigma" in
            publications, which determines the length-scale at which the short-range and
            long-range parts of the naive 1/r^p potential are separated. For the Coulomb
            potential (p=1), this potential can be interpreted as the effective
            potential generated by a Gaussian charge density, in which case this
            smearing parameter corresponds to the "width" of the Gaussian.
        """
        exponent = torch.tensor(self.exponent, device=k_sq.device, dtype=k_sq.dtype)
        peff = (3 - exponent) / 2
        prefac = (math.pi) ** 1.5 / gamma(exponent / 2) * (2 * smearing**2) ** peff
        x = 0.5 * smearing**2 * k_sq
        fourier = prefac * gammaincc(peff, x) / x**peff * gamma(peff)

        return fourier

    def potential_fourier_at_zero(self, smearing: float) -> torch.Tensor:
        """
        The value of the Fourier-transformed potential (LR part implemented above) as
        k --> 0 often needs to be set separately since for exponents p <= 3 = dimension,
        there is a divergence to +infinity.
        Setting this value manually to zero physically corresponds to the addition of a
        uniform backgruond charge to make the system charge-neutral.
        For p > 3, on the other hand, the Fourier-transformed LR potential does not
        diverge as k --> 0, and one could instead assign the correct limit.
        This is not implemented for now for consistency reasons.

        :param smearing: float containing the parameter often called "sigma" in
            publications, which determines the length-scale at which the short-range and
            long-range parts of the naive 1/r^p potential are separated. For the Coulomb
            potential (p=1), this potential can be interpreted as the effective
            potential generated by a Gaussian charge density, in which case this
            smearing parameter corresponds to the "width" of the Gaussian.
        """
        return torch.tensor(0.0)
