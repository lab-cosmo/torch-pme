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
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self.device = "cpu" if device is None else device
        if smearing is not None:
            self.register_buffer(
                "smearing", torch.tensor(smearing, device=self.device, dtype=self.dtype)
            )
        else:
            self.smearing = None
        if exclusion_radius is not None:
            self.register_buffer(
                "exclusion_radius",
                torch.tensor(exclusion_radius, device=self.device, dtype=self.dtype),
            )
        else:
            self.exclusion_radius = None

    @torch.jit.export
    def f_cutoff(self, vector: torch.Tensor) -> torch.Tensor:
        r"""TODO: Add docstring"""
        r_mag = torch.norm(vector, dim=1)
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
        return scalar_potential.unsqueeze(-1) * torch.eye(3).to(r_outer).unsqueeze(
            0
        ) - 3.0 * r_outer / (r_mag**5).unsqueeze(-1)

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
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )
        alpha = 1 / (2 * self.smearing**2)
        r_mag = torch.norm(vector, dim=1, keepdim=True)
        scalar_potential = (
            1.0 / (r_mag**3)
            - torch.erfc(torch.sqrt(alpha) * r_mag) / r_mag**3
            - 2 * torch.sqrt(alpha / torch.pi) * torch.exp(-alpha * r_mag**2) / r_mag**2
        )
        r_outer = torch.einsum("bi,bj->bij", vector, vector)
        return scalar_potential.unsqueeze(-1) * torch.eye(3).to(r_outer).unsqueeze(
            0
        ) - r_outer * (
            3.0 / (r_mag**5)
            - 3.0 * torch.erfc(torch.sqrt(alpha) * r_mag) / r_mag**5
            - 2
            * torch.sqrt(alpha / torch.pi)
            * (2 * alpha + 3 / r_mag**2)
            * torch.exp(-alpha * r_mag**2)
            / r_mag**2
        ).unsqueeze(-1)

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        r"""TODO: Add docstring"""
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range kernel without specifying `smearing`."
            )

        # avoid NaNs in backward, see
        # https://github.com/jax-ml/jax/issues/1052
        # https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
        masked = torch.where(k_sq == 0, 1.0, k_sq)
        return torch.where(
            k_sq == 0,
            0.0,
            4 * torch.pi * torch.exp(-0.5 * self.smearing**2 * masked) / masked,
        )

    def self_contribution(self) -> torch.Tensor:
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )
        alpha = 1 / (2 * self.smearing**2)
        return 4 * torch.pi / 3 * torch.sqrt((alpha / torch.pi) ** 3)

    def background_correction(self) -> torch.Tensor:
        if self.smearing is None:
            raise ValueError(
                "Cannot compute background correction without specifying `smearing`."
            )
        return self.smearing * 0.0

    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
