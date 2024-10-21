import math
from typing import Optional

import torch
from torch.special import gammainc, gammaincc, gammaln

from .splines import (
    CubicSpline,
    CubicSplineLongRange,
    compute_second_derivatives,
    compute_spline_ft,
)


class Potential(torch.nn.Module):
    r"""Base class defining the interface for a pair potential energy function

    The class provides the interface to compute a short-range and long-range functions
    in real space (such that :math:`V(r)=V_{\mathrm{SR}}(r)+V_{\mathrm{LR}}(r)` ), as
    well as a reciprocal-space version of the long-range component
    :math:`\hat{V}_{\mathrm{LR}}(k))` ).

    Derived classes can decide to implement a subset of these functionalities (e.g.
    providing only the real-space potential :math:`V(r)`). Internal state variables and
    parameters in derived classes should be defined in the ``__init__``  method.

    This base class also provides parameters to set the length scale associated with the
    range separation (``smearing``), and a cutoff function that can be optionally
    set to zero out the potential *inside* a short-range ``exclusion_radius``. This is
    often useful when combining ``torch-pme``-based ML models with local models that are
    better suited to describe the structure within a local cutoff.

    Note that a :py:class:`Potential` class can also be used inside a
    :py:class:`KSpaceFilter`, see :py:func:`Potential.kernel_from_k_sq`.

    :param smearing: The length scale associated with the switching between
        :math:`V_{\mathrm{SR}}(r)` and :math:`V_{\mathrm{LR}}(r)`
    :param exclusion_radius: A length scale that defines a *local environment* within which
        the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param dtype: Optional, the type used for the internal buffers and parameters
    :param device: Optional, the device used for the internal buffers and parameters
    """

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")
        if smearing is not None:
            self.register_buffer(
                "smearing", torch.tensor(smearing, device=device, dtype=dtype)
            )
        else:
            self.smearing = None
        if exclusion_radius is not None:
            self.register_buffer(
                "exclusion_radius",
                torch.tensor(exclusion_radius, device=device, dtype=dtype),
            )
        else:
            self.exclusion_radius = None

    @torch.jit.export
    def f_cutoff(self, dist: torch.Tensor) -> torch.Tensor:
        r"""
        Default cutoff function defining the *local* region that should be excluded from
        the computation of a long-range model. Defaults to a shifted cosine
        :math:`(1+\cos \pi r/r_\mathrm{cut})/2`.

        :param dist: a torc.Tensor containing the interatomic distances over which the
            cutoff function should be computed.
        """

        if self.exclusion_radius is None:
            raise ValueError(
                "Cannot compute cutoff function when `exclusion_radius` is not set"
            )

        return torch.where(
            dist < self.exclusion_radius,
            (1 + torch.cos(dist * (torch.pi / self.exclusion_radius))) * 0.5,
            0.0,
        )

    @torch.jit.export
    def from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """Computes a pair potential given a tensor of interatomic distances.

        :param dist: torch.Tensor containing the distances at which the potential
            is to be evaluated.
        """

        raise NotImplementedError(
            f"from_dist is not implemented for {self.__class__.__name__}"
        )

    @torch.jit.export
    def sr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        r"""Short-range part of the pair potential in real space.

        Even though one can provide a custom version, this is usually evaluated as
        :math:`V_{\mathrm{SR}}(r)=V(r)-V_{\mathrm{LR}}(r)`, based on the full and
        long-range parts of the potential. If the parameter ``exclusion_radius`` is
        defined, it computes this part as
        :math:`V_{\mathrm{SR}}(r)=-V_{\mathrm{LR}}(r)*f_\mathrm{cut}(r)` so that, when
        added to the part of the potential computed in the Fourier domain, the potential
        within the local region goes smoothly to zero.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """

        if self.smearing is None:
            raise ValueError(
                "Cannot compute range-separated potential when `smearing` is not specified."
            )
        if self.exclusion_radius is None:
            return self.from_dist(dist) - self.lr_from_dist(dist)
        return -self.lr_from_dist(dist) * self.f_cutoff(dist)

    @torch.jit.export
    def lr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the long-range part of the pair potential :math:`V_\mathrm{LR}(r)`. in
        real space, given a tensor of interatomic distances.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """

        raise NotImplementedError(
            f"lr_from_dist is not implemented for {self.__class__.__name__}"
        )

    @torch.jit.export
    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the Fourier-domain version of the long-range part of the pair potential
        :math:`\hat{V}_\mathrm{LR}(k)`. The function is expressed in terms of
        :math:`k^2`, as that avoids, in several important cases, an unnecessary square
        root operation.

        :param k_sq: torch.tensor containing the squared norm of the Fourier domain
            vectors at which :math:`\hat{V}_\mathrm{LR}` must be evaluated.
        """
        raise NotImplementedError(
            f"lr_from_k_sq is not implemented for {self.__class__.__name__}"
        )

    @torch.jit.export
    def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """
        Compatibility function with the interface of :py:class:`KSpaceKernel`, so that
        potentials can be used as kernels for :py:class:`KSpaceFilter`.
        """

        return self.lr_from_k_sq(k_sq)

    @torch.jit.export
    def self_contribution(self) -> torch.Tensor:
        """
        A correction that depends exclusively on the "charge" on every particle and on
        the range splitting parameter. Foe example, in the case of a Coulomb potential,
        this is the potential generated at the origin by the fictituous Gaussian charge
        density in order to split the potential into a SR and LR part.
        """
        raise NotImplementedError(
            f"self_contribution is not implemented for {self.__class__.__name__}"
        )

    @torch.jit.export
    def background_correction(self) -> torch.Tensor:
        """
        A correction designed to compensate for the presence of divergent terms. For
        instance, the energy of a periodic electrostatic system is infinite when the
        cell is not charge-neutral. This term then implicitly assumes that a homogeneous
        background charge of the opposite sign is present to make the cell neutral.
        """
        raise NotImplementedError(
            f"background_correction is not implemented for {self.__class__.__name__}"
        )


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
        LR part of the range-separated :math:`1/r` potential.

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


# since pytorch has implemented the incomplete Gamma functions, but not the much more
# commonly used (complete) Gamma function, we define it in a custom way to make autograd
# work as in https://discuss.pytorch.org/t/is-there-a-gamma-function-in-pytorch/17122
def gamma(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(gammaln(x))


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
        :py:class:`Potential`.
    """

    def __init__(
        self,
        exponent: float,
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

        if exponent <= 0 or exponent > 3:
            raise ValueError(f"`exponent` p={exponent} has to satisfy 0 < p <= 3")
        self.register_buffer(
            "exponent", torch.tensor(exponent, dtype=dtype, device=device)
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
        """
        Fourier transform of the LR part potential in terms of :math:`k^2`.

        If only the Coulomb potential is needed, the last line can be
        replaced by

        .. code-block:: python

            fourier = 4 * torch.pi * torch.exp(-0.5 * smearing**2 * k_sq) / k_sq

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


class SplinePotential(Potential):
    """
    Potential built from a spline interpolation.

    Here :math:`r` is a distance parameter and :math:`p` an exponent.

    The short and long-range parts of the potential are computed based on a cubic
    spline built at initialization time. The reciprocal-space kernel is computed
    numerically as a spline, too.  Assumes the infinite-separation value of the
    potential to be zero.
    """

    def __init__(
        self,
        r_grid: torch.Tensor,
        y_grid: Optional[torch.Tensor] = None,
        r_grid_lr: Optional[torch.Tensor] = None,
        y_grid_lr: Optional[torch.Tensor] = None,
        k_grid: Optional[torch.Tensor] = None,
        krn_grid: Optional[torch.Tensor] = None,
        smearing: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing=smearing, dtype=dtype, device=device)
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")

        if r_grid_lr is None:
            r_grid_lr = r_grid

        if k_grid is None:  # defaults to 1/lr_grid_points
            k_grid = torch.pi * 2 * torch.reciprocal(r_grid_lr).flip(dims=[0])

        if y_grid is None:
            self._spline = None
        else:
            self._spline = CubicSpline(r_grid, y_grid)

        if y_grid_lr is None:
            self._lr_spline = None
        else:
            self._lr_spline = CubicSplineLongRange(r_grid_lr, y_grid_lr)

        if krn_grid is None and y_grid_lr is not None:
            # computes automatically!
            krn_grid = compute_spline_ft(
                k_grid,
                r_grid_lr,
                y_grid_lr,
                compute_second_derivatives(r_grid_lr, y_grid_lr),
            )

        if krn_grid is None:
            self._krn_spline = None
        else:
            self._krn_spline = CubicSpline(k_grid**2, krn_grid)

    def from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Full potential as a function of :math:`r`.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """

        # if the full spline is not given, falls back on the lr part
        if self._spline is None:
            return self.lr_from_dist(dist)
        return self._spline(dist)

    def sr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        LR part of the range-separated potential.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """

        return self.from_dist(dist) - self.lr_from_dist(dist)

    def lr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        LR part of the range-separated potential.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """

        if self._lr_spline is None:
            return 0.0 * dist
        return self._lr_spline(dist)

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """
        Fourier transform of the LR part potential in terms of :math:`k^2`.

        :param k_sq: torch.tensor containing the squared lengths (2-norms) of the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        """

        if self._krn_spline is None:
            return k_sq * 0.0
        return self._krn_spline(k_sq)

    def self_contribution(self) -> torch.Tensor:
        # self-correction for 1/r potential

        return torch.tensor([0.0])

    def background_correction(self) -> torch.Tensor:
        # "charge neutrality" correction for 1/r potential
        return torch.tensor([0.0])
