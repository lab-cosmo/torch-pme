from typing import Optional

import torch

from .potential import Potential


class PotentialDipole(Potential):
    """TODO: Add docstring"""

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

    def from_dist(self, vector: torch.Tensor) -> torch.Tensor:
        """TODO: Add docstring"""
        r_mag = torch.norm(vector, dim=1, keepdim=True)
        scalar_potential = 1.0 / (r_mag**3)
        r_outer = torch.einsum(
            "bi,bj->bij", vector, vector
        )  # outer product shape (batch, 3, 3)
        tensor_potential = (3.0 / (r_mag**5)).unsqueeze(-1) * r_outer
        return scalar_potential, tensor_potential

    def lr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: Implement smearing for `lr_from_dist`")

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: Implement smearing for `lr_from_k_sq`")

    def self_contribution(self) -> torch.Tensor:
        raise NotImplementedError("TODO: Implement smearing for `self_contribution`")

    def background_correction(self) -> torch.Tensor:
        raise NotImplementedError(
            "TODO: Implement smearing for `background_correction`"
        )

    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
