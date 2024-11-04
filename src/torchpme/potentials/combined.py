from typing import Optional

import torch

from .potential import Potential


class CombinedPotential(Potential):
    """
    A potential that is a linear combination of multiple potentials.

    A class representing a combined potential that aggregates multiple individual
    potentials with weights for use in long-range (LR) and short-range (SR)
    interactions.

    The ``CombinedPotential`` class allows for flexible combination of potential
    functions with user-specified weights, which can be either fixed or trainable.

    :param potentials: List of potential objects, each implementing a compatible
        interface with methods `from_dist`, `lr_from_dist`, `lr_from_k_sq`,
        `self_contribution`, and `background_correction`.
    :param initial_weights: Initial weights for combining the potentials. If provided,
        the length must match the number of potentials. If `None`, weights are
        initialized to ones.
    :param learnable_weights: If `True`, weights are trainable parameters, allowing
        optimization during training. If `False`, weights are fixed.
    :param exclusion_radius: A length scale that defines a *local environment* within
        which the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param dtype: type used for the internal buffers and parameters
    :param device: device used for the internal buffers and parameters
    """

    def __init__(
        self,
        potentials: list[Potential],
        initial_weights: Optional[torch.Tensor] = None,
        learnable_weights: Optional[bool] = True,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            smearing=smearing,
            exclusion_radius=exclusion_radius,
            dtype=dtype,
            device=device,
        )
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")
        smearings = [pot.smearing for pot in potentials]
        if not all(smearings) and any(smearings):
            raise ValueError(
                r"Cannot combine direct (`smearing=None`) and range-separated (`smearing=float`) potentials."
            )

        if all(smearings) and not self.smearing:
            # this is very misleading, but it is the way the original code works,
            # otherwise mypy complains
            raise ValueError(
                r"You should specify a `smearing` when combining range-separated (`smearing=float`) potentials."
            )
        if not any(smearings) and self.smearing:
            # this is very misleading, but it is the way the original code works,
            # otherwise mypy complai
            raise ValueError(
                r"Cannot specify `smearing` when combining direct (`smearing=None`) potentials."
            )

        if initial_weights is not None:
            if len(initial_weights) != len(potentials):
                raise ValueError(
                    "The number of initial weights must match the number of potentials being combined"
                )
        else:
            initial_weights = torch.ones(len(potentials), dtype=dtype, device=device)
        # for torchscript
        self.potentials = torch.nn.ModuleList(potentials)
        if learnable_weights:
            self.weights = torch.nn.Parameter(initial_weights)
        else:
            self.register_buffer("weights", initial_weights)

    def from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        potentials = [pot.from_dist(dist) for pot in self.potentials]
        potentials = torch.stack(potentials, dim=-1)
        return torch.inner(self.weights, potentials)

    def sr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        potentials = [pot.sr_from_dist(dist) for pot in self.potentials]
        potentials = torch.stack(potentials, dim=-1)
        return torch.inner(self.weights, potentials)

    def lr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        potentials = [pot.lr_from_dist(dist) for pot in self.potentials]
        potentials = torch.stack(potentials, dim=-1)
        return torch.inner(self.weights, potentials)

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        potentials = [pot.lr_from_k_sq(k_sq) for pot in self.potentials]
        potentials = torch.stack(potentials, dim=-1)
        return torch.inner(self.weights, potentials)

    def self_contribution(self) -> torch.Tensor:
        # self-correction for 1/r^p potential
        potentials = [pot.self_contribution() for pot in self.potentials]
        potentials = torch.stack(potentials, dim=-1)
        return torch.inner(self.weights, potentials)

    def background_correction(self) -> torch.Tensor:
        # "charge neutrality" correction for 1/r^p potential
        potentials = [pot.background_correction() for pot in self.potentials]
        potentials = torch.stack(potentials, dim=-1)
        return torch.inner(self.weights, potentials)

    from_dist.__doc__ = Potential.from_dist.__doc__
    sr_from_dist.__doc__ = Potential.sr_from_dist.__doc__
    lr_from_dist.__doc__ = Potential.lr_from_dist.__doc__
    lr_from_k_sq.__doc__ = Potential.lr_from_k_sq.__doc__
    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
