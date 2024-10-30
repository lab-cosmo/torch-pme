from typing import Optional

import torch

from .coulomb import CoulombPotential


class P3MCoulombPotential(CoulombPotential):
    r"""Coulomb potential for the P3M method."""

    def __init__(
        self,
        interpolation_nodes: int,
        mesh_spacing: torch.Tensor,
        cell: torch.Tensor,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)
        self.interpolation_nodes = interpolation_nodes
        self.mesh_spacing = mesh_spacing.reshape(1, 1, 1, 1, 3)
        self.cell = cell
        inverse_cell = torch.linalg.inv(cell)
        self._reciprocal_cell = 2 * torch.pi * inverse_cell.T

    @cached_property
    def _volume(self) -> torch.Tensor:
        """Volume of one mesh cell"""
        return torch.det(self.cell * self.mesh_spacing.reshape(3, 1))

    @cached_property
    def _k_vectors(self) -> torch.Tensor:
        return torch.concat(
            [torch.zeros((1, 3)), self._reciprocal_cell, -self._reciprocal_cell[:-1]],
            dim=0,
        )[
            torch.newaxis, torch.newaxis, torch.newaxis, ...
        ]  # Shape (1, 1, 1, 6, 3), no -z direction vector

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

        k = k.double()
        print(f"{k.shape=}")
        D = self._diff_operator(k)
        print(f"{D.shape=}")
        U2 = (
            self._charge_assignment(torch.unsqueeze(k, -2) + self._k_vectors)
            / self._volume
        ) ** 2
        print(f"{U2.shape=}")
        R = self._reference_force(torch.unsqueeze(k, -2) + self._k_vectors)
        print(f"{R.shape=}")
        D2 = torch.linalg.norm(D, dim=-1) ** 2
        print(f"{D2.shape=}")
        denominator = D2 * torch.sum(U2, dim=-1) ** 2
        print(f"{denominator.shape=}")

        return torch.where(
            denominator == 0,
            0.0j,
            torch.squeeze(
                torch.unsqueeze(D, dim=-2)  # (nx, ny, nz, 1, 3)
                @ torch.unsqueeze(
                    torch.sum(
                        torch.unsqueeze(U2, -1) * R,  # (nx, ny, nz, 67, 3)
                        dim=-2,
                    ),  # (nx, ny, nz, 3)
                    dim=-1,
                ),  # (nx, ny, nz, 3, 1) -> (nx, ny, nz, 1, 1)
                dim=[-2, -1],
            )  # (nx, ny, nz)
            / denominator,
        )

    def _diff_operator(self, k: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, 3)"""
        return (
            1j
            * torch.sin(k * self.mesh_spacing.squeeze(0))
            / (self.mesh_spacing.squeeze(0))
        )

    def _charge_assignment(self, k: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 6, 3) to shape (nx, ny, nz, 6)"""
        return torch.prod(
            self.mesh_spacing
            * torch.sinc(k * self.mesh_spacing / 2 / torch.pi)
            ** self.interpolation_nodes,
            dim=-1,
        )

    def _reference_force(self, k: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 6, 3) to shape (nx, ny, nz, 6, 3)"""

        k_sq = torch.linalg.norm(k, dim=-1) ** 2
        print(f"{k_sq.shape=}")

        return (
            k
            * torch.where(
                k_sq == 0,
                0.0j,
                -1j * 4 * torch.pi * torch.exp(-0.5 * self.smearing**2 * k_sq) / k_sq,
            )[..., torch.newaxis]
        )
