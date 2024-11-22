from typing import Optional

import torch


class Potential(torch.nn.Module):
    r"""
    Base class defining the interface for a pair potential energy function

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

    Note that a :class:`Potential` class can also be used inside a
    :class:`KSpaceFilter`, see :func:`Potential.kernel_from_k_sq`.

    :param smearing: The length scale associated with the switching between
        :math:`V_{\mathrm{SR}}(r)` and :math:`V_{\mathrm{LR}}(r)`
    :param exclusion_radius: A length scale that defines a *local environment* within
        which the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param dtype: type used for the internal buffers and parameters
    :param device: device used for the internal buffers and parameters
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
        """
        Computes a pair potential given a tensor of interatomic distances.

        :param dist: torch.Tensor containing the distances at which the potential
            is to be evaluated.
        """
        raise NotImplementedError(
            f"from_dist is not implemented for {self.__class__.__name__}"
        )

    @torch.jit.export
    def sr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        r"""
        Short-range part of the pair potential in real space.

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
        Compatibility function with the interface of :class:`KSpaceKernel`, so that
        potentials can be used as kernels for :class:`KSpaceFilter`.
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
