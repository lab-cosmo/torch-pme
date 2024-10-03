import math
from typing import Optional, Union

import torch
from torch.special import gammainc, gammaincc, gammaln

# TODO MUST POLISH DOCUMENTATION AND REFACTOR TESTS

class Potential(torch.nn.Module):
    r"""
    Base class defining the interface for a pair potential energy function

    It provides the interface to compute a short-range and long-range
    functions in real space (such that
    :math:`V(r)=V_{\mathrm{SR}}(r)+V_{\mathrm{LR}}(r)` ),
    as well as a reciprocal-space version of the long-range
    component :math:`\hat{V}_{\mathrm{LR}}(k))` ).
    
    Derived classes can decide to implement a subset of these 
    functionalities (e.g. providing only the real-space potential
    :math:`V(r)`). 
    Internal state variables and parameters in derived classes should 
    be defined in the ``__init__``  method. 

    This base class also provides parameters to set the length
    scale associated with the range separation, and a cutoff 
    function that can be optionally set to zero out the potential
    *inside* a short-range cutoff. This is often useful when
    combining ``torch-pme``-based ML models with local models that
    are better suited to describe the structure within a local
    cutoff.

    Note that a :py:class:`Potential` class can also be used 
    inside a :py:class:`KSpaceFilter`, see 
    :py:func:`Potential.kernel_from_k_sq`.

    :param range_radius: The length scale associated with the 
        switching between 
        :math:`V_{\mathrm{SR}}(r)` and :math:`V_{\mathrm{LR}}(r)`        
    :param cutoff_radius: A length scale that defines a 
        *local environment* within which the potential should be 
        smoothly zeroed out, as it will be described by a separate
        model.        
    """

    def __init__(
        self,
        range_radius: Optional[float] = None,
        cutoff_radius: Optional[float] = None,
    ):
        super().__init__()
        self.range_radius = range_radius
        self.cutoff_radius = cutoff_radius

    @torch.jit.export
    def f_cutoff(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Default cutoff function defining the *local* region
        that should be excluded from the computation of a
        long-range model. Defaults to a shifted cosine
        :math:`(1+\cos \pi r/r_\mathrm{cut})/2`.

        :param dist: a torc.Tensor containing the interatomic
            distances over which the cutoff function should be
            computed. 
        """

        if self.cutoff_radius is None:
            raise ValueError(
                "Cannot compute cutoff function when cutoff radius is not set"
            )
        return torch.where(
            dist < self.cutoff_radius,
            (1 + torch.cos(dist * (torch.pi / self.cutoff_radius))) * 0.5,
            0.0,
        )

    @torch.jit.export
    def from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Computes a pair potential given a tensor of interatomic distances.

        :param dist: torch.tensor containing the distances at which the potential
            is to be evaluated.
        """

        raise NotImplementedError(
            f"from_dist is not implemented for {self.__class__.__name__}"
        )

    @torch.jit.export
    def sr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the short-range part of the pair potential 
        in real space, given a tensor of interatomic distances.
        Even though one can provide a custom version, this is usually
        evaluated as 
        :math:`V_{\mathrm{SR}}(r)=V(r)-V_{\mathrm{LR}}(r)`, 
        based on the full and long-range parts of the potential.
        If the parameter ``cutoff_radius`` is defined, it computes 
        this part as 
        :math:`V_{\mathrm{SR}}(r)=-V_{\mathrm{LR}}(r)*f_\mathrm{cut}(r)`
        so that, when added to the part of the potential computed
        in the Fourier domain, the potential within the local region
        goes smoothly to zero.

        :param dist: torch.tensor containing the distances at which the potential
            is to be evaluated.
        """

        if self.cutoff_radius is None:
            return self.from_dist(dist) - self.lr_from_dist(dist)
        return -self.lr_from_dist(dist) * self.f_cutoff(dist)

    @torch.jit.export
    def lr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the long-range part of the pair potential 
        :math:`V_\mathrm{LR}(r)`.
        in real space, given a tensor of interatomic distances.
        
        :param dist: torch.tensor containing the distances at which the potential
            is to be evaluated.
        """

        raise NotImplementedError(
            f"lr_from_dist is not implemented for {self.__class__.__name__}"
        )

    @torch.jit.export
    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """
        Computes the Fourier-domain version of the long-range part of the pair potential 
        :math:`\hat{V}_\mathrm{LR}(k)`. The function is expressed in terms of 
        :math:`k^2`, as that avoids, in several important cases, an 
        unnecessary square root operation. 

        :param k_sq: torch.tensor containing the squared norm of the 
            Fourier domain vectors at which :math:`\hat{V}_\mathrm{LR}`
            must be evaluated.
        """
        raise NotImplementedError(
            f"lr_from_k_sq is not implemented for {self.__class__.__name__}"
        )

    @torch.jit.export
    def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """
        Compatibility function with the interface of
        :py:class:`KSpaceKernel`, so that potentials can be
        used as kernels for :py:class:`KSpaceFilter`.
        """

        return self.lr_from_k_sq(k_sq)


class InversePowerLawPotential(Potential):
    """
    Inverse power-law potentials of the form :math:`1/r^p`.

    Herem :math:`r` is a distance parameter and :math:`p` an exponent.

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
    """

    def __init__(
        self,
        exponent: float,
        range_radius: Union[float, torch.Tensor],
        cutoff_radius: Union[float, torch.Tensor],
    ):
        super().__init__(range_radius, cutoff_radius)
        self.exponent = exponent

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
        LR part of the range-separated :math:`1/r^p` potential.

        Used to subtract out the interior contributions after computing the LR part in
        reciprocal (Fourier) space.

        For the Coulomb potential, this would return (note that the only change between
        the SR and LR parts is the fact that erfc changes to erf)

        .. code-block:: python

            potential = erf(dist / sqrt(2) / smearing) / dist

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """

        exponent = torch.full([], self.exponent, device=dist.device, dtype=dist.dtype)
        smearing = torch.full(
            [], self.range_radius, device=dist.device, dtype=dist.dtype
        )
        x = 0.5 * dist**2 / smearing**2
        peff = exponent / 2
        prefac = 1.0 / (2 * smearing**2) ** peff
        return prefac * gammainc(peff, x) / x**peff

    @torch.jit.export
    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """
        Fourier transform of the LR part potential in terms of :math:`k^2`.

        If only the Coulomb potential is needed, the last line can be
        replaced by

        .. code-block:: python

            fourier = 4 * torch.pi * torch.exp(-0.5 * smearing**2 * k_sq) / k_sq

        :param k_sq: torch.tensor containing the squared lengths (2-norms) of the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        :param smearing: float containing the parameter often called "sigma" in
            publications, which determines the length-scale at which the short-range and
            long-range parts of the naive :math:`1/r^p` potential are separated. For the
            Coulomb potential (:math:`p=1`), this potential can be interpreted as the
            effective potential generated by a Gaussian charge density, in which case
            this smearing parameter corresponds to the "width" of the Gaussian.
        """
        exponent = torch.full([], self.exponent, device=k_sq.device, dtype=k_sq.dtype)
        smearing = torch.full(
            [], self.range_radius, device=k_sq.device, dtype=k_sq.dtype
        )
        peff = (3 - exponent) / 2
        prefac = math.pi**1.5 / gamma(exponent / 2) * (2 * smearing**2) ** peff
        x = 0.5 * smearing**2 * k_sq

        # The k=0 term often needs to be set separately since for exponents p<=3
        # dimension, there is a divergence to +infinity. Setting this value manually
        # to zero physically corresponds to the addition of a uniform background charge
        # to make the system charge-neutral. For p>3, on the other hand, the
        # Fourier-transformed LR potential does not diverge as k->0, and one
        # could instead assign the correct limit. This is not implemented for now
        # for consistency reasons.
        return torch.where(
            k_sq == 0,
            0.0,
            prefac * gammaincc(peff, x) / x**peff * gamma(peff),
        )


# since pytorch has implemented the incomplete Gamma functions, but not the much more
# commonly used (complete) Gamma function, we define it in a custom way to make autograd
# work as in https://discuss.pytorch.org/t/is-there-a-gamma-function-in-pytorch/17122
def gamma(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(gammaln(x))
