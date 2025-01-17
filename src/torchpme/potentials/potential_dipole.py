from typing import Optional

import torch

from .potential import Potential


class PotentialDipole(torch.nn.Module):
    """TODO: Add docstring"""

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
    def f_cutoff(self, vector: torch.Tensor) -> torch.Tensor:
        r"""TODO: Add docstring"""
        r_mag = torch.norm(vector, dim=1, keepdim=True)
        if self.exclusion_radius is None:
            raise ValueError(
                "Cannot compute cutoff function when `exclusion_radius` is not set"
            )

        return torch.where(
            r_mag < self.exclusion_radius,
            (1 + torch.cos(r_mag * (torch.pi / self.exclusion_radius))) * 0.5,
            0.0,
        )

    def from_dist(self, vector: torch.Tensor) -> torch.Tensor:
        """TODO: Add docstring"""
        r_mag = torch.norm(vector, dim=1, keepdim=True)
        scalar_potential = 1.0 / (r_mag**3)
        r_outer = torch.einsum(
            "bi,bj->bij", vector, vector
        )  # outer product shape (batch, 3, 3)
        tensor_potential = (3.0 / (r_mag**5)).unsqueeze(-1) * r_outer
        return scalar_potential, tensor_potential
    
    @torch.jit.export
    def sr_from_dist(self, vector: torch.Tensor) -> torch.Tensor:
        r"""TODO: Add docstring"""
        if self.smearing is None:
            raise ValueError(
                "Cannot compute range-separated potential when `smearing` is not specified."
            )
        if self.exclusion_radius is None:
            return self.from_dist(vector) - self.lr_from_dist(vector)
        return -self.lr_from_dist(vector) * self.f_cutoff(vector)
    
    @torch.jit.export
    def lr_from_dist(self, vector: torch.Tensor) -> torch.Tensor:
        r"""TODO: Add docstring"""

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
