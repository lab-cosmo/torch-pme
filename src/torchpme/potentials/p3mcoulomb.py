from typing import Optional

import torch

from .coulomb import CoulombPotential


class P3MCoulombPotential(CoulombPotential):
    r"""Coulomb potential for the P3M method."""

    def __init__(
        self,
        interpolation_nodes: int,
        mesh_spacing: float,
        cell: torch.Tensor,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)
        self.interpolation_nodes = interpolation_nodes
        self.mesh_spacing = mesh_spacing
        self.cell = cell

    @cached_property
    def _volume(self) -> torch.Tensor:
        return torch.det(self.cell)

    @cached_property
    def _k_vectors(self) -> torch.Tensor:
        inverse_cell = torch.linalg.inv(self.cell)
        reciprocal_cell = 2 * torch.pi * inverse_cell.T
        return torch.concat(
            [torch.zeros((1, 3)), reciprocal_cell, -reciprocal_cell], dim=0
        )[torch.newaxis, torch.newaxis, torch.newaxis, ...]  # Shape (1, 1, 1, 7, 3)

    @torch.jit.export
    def kernel_from_k(self, k: torch.Tensor) -> torch.Tensor:
        """
        Compatibility function with the interface of :py:class:`KSpaceKernel`, so that
        potentials can be used as kernels for :py:class:`KSpaceFilter`.
        """

        return self.lr_from_k(k)

    def lr_from_k(self, k: torch.Tensor) -> torch.Tensor:
        """
        Fourier transform of the LR part potential in terms of :math:`k`.

        :param k: torch.tensor containing the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        """

        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range kernel without specifying `smearing`."
            )

        diff_operator = self._diff_operator(k)
        charge_assignment = (
            self._charge_assignment(torch.unsqueeze(k, -2) + self._k_vectors)
            / self._volume
        ) ** 2
        print(charge_assignment.shape)
        reference_force = self._reference_force(
            torch.unsqueeze(k, -2) + self._k_vectors
        )
        print(reference_force.shape)
        diff_operator_sq = torch.linalg.norm(diff_operator, dim=-1) ** 2

        return (
            torch.squeeze(
                torch.unsqueeze(diff_operator, dim=-2)  # (nx, ny, nz, 1, 3)
                @ torch.unsqueeze(
                    torch.sum(
                        torch.unsqueeze(charge_assignment, -1)
                        * reference_force,  # (nx, ny, nz, 7, 3)
                        dim=-2,
                    ),  # (nx, ny, nz, 3)
                    dim=-1,
                ),  # (nx, ny, nz, 3, 1) -> (nx, ny, nz, 1, 1)
                dim=[-2, -1],
            )  # (nx, ny, nz)
            / diff_operator_sq
            / torch.sum(charge_assignment) ** 2
        )

    def _diff_operator(self, k: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, 3)"""
        return 1j * torch.sin(k * self.mesh_spacing) / (self.mesh_spacing)

    def _charge_assignment(self, k: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 7, 3) to shape (nx, ny, nz, 7)"""
        return torch.prod(
            self.mesh_spacing
            * (torch.sin(k * self.mesh_spacing / 2) / (k * self.mesh_spacing / 2))
            ** self.interpolation_nodes,
            dim=-1,
        )

    def _reference_force(self, k: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 7, 3) to shape (nx, ny, nz, 7, 3)"""
        return -1j * k * 4 * torch.pi / k**2 * torch.exp(-0.5 * self.smearing**2 * k**2)
