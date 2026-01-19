from typing import Optional

import torch


class Potential(torch.nn.Module):
    r"""
    Base interface for a pair potential energy function between monopoles.

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
    :param exclusion_degree: Controls the sharpness of the transition in the cutoff function
        applied within the ``exclusion_radius``. The cutoff is computed as a raised cosine
        with exponent ``exclusion_degree``
    :param prefactor: potential prefactor; see :ref:`prefactors` for details and common
        values of electrostatic prefactors.
    """

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
        prefactor: float = 1.0,
    ):
        super().__init__()

        if smearing is not None:
            self.register_buffer(
                "smearing", torch.tensor(smearing, dtype=torch.float64)
            )
        else:
            self.smearing = None

        self.exclusion_radius = exclusion_radius
        self.exclusion_degree = exclusion_degree
        self.register_buffer("prefactor", torch.tensor(prefactor, dtype=torch.float64))

    @torch.jit.export
    def f_cutoff(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Default cutoff function defining the *local* region that should be excluded from
        the computation of a long-range model. Defaults to a shifted cosine
        :math:`1 - ((1 - \cos \pi r/r_\mathrm{cut})/2) ^ n`. where :math:`n` is the
        ``exclusion_degree`` parameter.

        :param dist: a torc.Tensor containing the interatomic distances over which the
            cutoff function should be computed.
        :param pair_mask: Optional torch.tensor containing a mask to be applied to the
            result.
        """
        if self.exclusion_radius is None:
            raise ValueError(
                "Cannot compute cutoff function when `exclusion_radius` is not set"
            )

        result = torch.where(
            dist < self.exclusion_radius,
            1
            - ((1 - torch.cos(torch.pi * (dist / self.exclusion_radius))) * 0.5)
            ** self.exclusion_degree,
            0.0,
        )
        if pair_mask is not None:
            result = result * pair_mask

        return result

    @torch.jit.export
    def from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes a pair potential given a tensor of interatomic distances.

        :param dist: torch.tensor containing the distances at which the potential
            is to be evaluated.
        :param pair_mask: Optional torch.tensor containing a mask to be applied to the
            result.
        """
        raise NotImplementedError(
            f"from_dist is not implemented for {self.__class__.__name__}"
        )

    @torch.jit.export
    def sr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Short-range (SR) part of the pair potential in real space.

        Even though one can provide a custom version, this is usually evaluated as
        :math:`V_{\mathrm{SR}}(r)=V(r)-V_{\mathrm{LR}}(r)`, based on the full and
        long-range parts of the potential. If the parameter ``exclusion_radius`` is
        defined, it computes this part as
        :math:`V_{\mathrm{SR}}(r)=-V_{\mathrm{LR}}(r)*f_\mathrm{cut}(r)` so that, when
        added to the part of the potential computed in the Fourier domain, the potential
        within the local region goes smoothly to zero.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        :param pair_mask: Optional torch.tensor containing a mask to be applied to the
            result.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute range-separated potential when `smearing` is not specified."
            )

        # Don't apply prefactor here as it is applied in child classes
        if self.exclusion_radius is None:
            return self.from_dist(dist, pair_mask=pair_mask) - self.lr_from_dist(
                dist, pair_mask=pair_mask
            )
        return -self.lr_from_dist(dist, pair_mask=pair_mask) * self.f_cutoff(
            dist, pair_mask=pair_mask
        )

    @torch.jit.export
    def lr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Computes the long-range part of the pair potential :math:`V_\mathrm{LR}(r)`. in
        real space, given a tensor of interatomic distances.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        :param pair_mask: Optional torch.tensor containing a mask to be applied to the
            result.
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

    @torch.jit.export
    def pbc_correction(
        self,
        periodic: Optional[torch.Tensor],
        positions: torch.Tensor,
        cell: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """A correction term that is only relevant for systems with 2D periodicity."""
        return self.prefactor * torch.zeros_like(charges)
