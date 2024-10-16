from typing import Optional

import torch

from .coulomb import CoulombPotential


class P3MCoulombPotential(CoulombPotential):

    def __init__(
        self,
        interpolation_nodes: int,
        mesh_spacing: int,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)
        self.interpolation_nodes = interpolation_nodes

    def lr_from_k(self, k_sq: torch.Tensor) -> torch.Tensor:
        """
        Fourier transform of the LR part potential in terms of :math:`k`.

        :param k_sq: torch.tensor containing the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        """

        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range kernel without specifying `smearing`."
            )

        pass

    def _diff_operator(self, k: torch.Tensor) -> torch.Tensor:
        pass

    def _charge_assignment(self, k: torch.Tensor) -> torch.Tensor:
        return torch.prod()

    def _reference_force(self, k: torch.Tensor) -> torch.Tensor:
        return -1j * k * 4 * torch.pi / k**2 * torch.exp(-0.5 * self.smearing**2 * k**2)

