from functools import cached_property
from typing import Optional

import torch
from torch.nn.functional import pad
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
        diff_order: Optional[int] = 2,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)
        self.interpolation_nodes = interpolation_nodes
        self.mesh_spacing = mesh_spacing.reshape(1, 1, 1, 3)
        self.cell = cell
        self.diff_order = diff_order
        if cell.is_cuda:
            # use function that does not synchronize with the CPU
            inverse_cell = torch.linalg.inv_ex(cell)[0]
        else:
            inverse_cell = torch.linalg.inv(cell)
        self._reciprocal_cell = 2 * torch.pi * inverse_cell.T

    @cached_property
    def _volume(self) -> torch.Tensor:
        """Volume of one mesh cell"""
        return torch.det(self.cell * self.mesh_spacing.reshape(3, 1))

    @cached_property
    def _k_vectors(self) -> torch.Tensor:
        m = torch.tensor([-2, -1, 0, 1, 2], dtype=self.cell.dtype).unsqueeze(-1)
        m = (
            pad(input=m, pad=(0, 2))[:, None, None]
            + pad(input=m, pad=(1, 1))[None, :, None]
            + pad(input=m, pad=(2, 0))[None, None, 2:]  # no -z direction vector
        ).reshape(-1, 3)
        m = m[torch.linalg.norm(m, dim=1) <= 2]
        return (m @ self._reciprocal_cell).reshape(1, 1, 1, -1, 3)

    @cached_property
    def _directions(self) -> torch.Tensor:
        m = torch.tensor([-2, -1, 0, 1, 2], dtype=self.cell.dtype).unsqueeze(-1)
        m = (
            pad(input=m, pad=(0, 2))[:, None, None]
            + pad(input=m, pad=(1, 1))[None, :, None]
            + pad(input=m, pad=(2, 0))[None, None, 2:]  # no -z direction vector
        ).reshape(-1, 3)
        return m[torch.linalg.norm(m, dim=1) <= 2].int().numpy()

    @torch.jit.export
    def kernel_from_k(self, k: torch.Tensor) -> torch.Tensor:
        """
        Compatibility function with the interface of :py:class:`KSpaceKernel`, so that
        potentials can be used as kernels for :py:class:`KSpaceFilter`.
        """
        result = self.lr_from_k(k)
        # print(result)
        return result

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

        kh = k.double() * self.mesh_spacing
        D = self._diff_operator(kh)
        # k_prime = torch.unsqueeze(k, -2) + self._k_vectors
        U2 = self._charge_assignment(kh) ** 2
        R = self._reference_force(k)
        D2 = torch.linalg.norm(D, dim=-1) ** 2
        numerator = torch.squeeze(
            torch.unsqueeze(D, dim=-2)  # (nx, ny, nz, 1, 3)
            @ torch.unsqueeze(
                torch.sum(
                    torch.unsqueeze(U2, -1) * R,  # (nx, ny, nz, 6, 3)
                    dim=-2,
                ),  # (nx, ny, nz, 3)
                dim=-1,
            ),  # (nx, ny, nz, 3, 1) -> (nx, ny, nz, 1, 1)
            dim=[-2, -1],
        )
        denominator = D2 * torch.sum(U2, dim=-1) ** 2

        print(f"{numerator = }")
        print(f"{denominator = }")

        return torch.where(
            denominator == 0,
            0.0,
            numerator / denominator,
        )

        # k = k.double()
        # print(f"{k.shape=}")
        # D = self._diff_operator(k)
        # print(f"{D.shape=}")
        # ks = torch.unsqueeze(k, -2) + self._k_vectors
        # U2 = (self._charge_assignment(ks) / self._volume) ** 2
        # print(f"{U2.shape=}")
        # DU2 = D.unsqueeze(-2) * U2.unsqueeze(-1)
        # print(f"{DU2.shape=}")
        # R = self._reference_force(ks)
        # print(f"{R.shape=}")
        # denominator = torch.sum(torch.linalg.norm(DU2, dim=-1), dim=-1) # ** 2
        # print(f"{denominator.shape=}")

        # return torch.where(
        #     denominator == 0,
        #     0.0j,
        #     torch.sum(
        #         DU2 * R,  # (nx, ny, nz, 6, 3)
        #         dim=(-2, -1),
        #     )  # (nx, ny, nz)
        #     / denominator,
        # )

    def _diff_operator(self, kh: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, 3)"""
        COEF = [
            [1],
            [4 / 3, -1 / 3],
            [3 / 2, -3 / 5, 1 / 10],
            [8 / 5, -4 / 5, 8 / 35, -1 / 35],
            [5 / 3, -20 / 21, 5 / 14, -5 / 63, 1 / 126],
            [12 / 7, -15 / 14, 10 / 21, -1 / 7, 2 / 77, -1 / 465],
        ]
        # kh = k * self.mesh_spacing
        temp = torch.zeros(kh.shape, dtype=kh.dtype, device=kh.device)
        for i, coef in enumerate(COEF[self.diff_order - 1]):
            temp += (coef / (i + 1)) * torch.sin(kh * (i + 1))
        return temp / (self.mesh_spacing)

    def _charge_assignment(self, kh: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd)"""
        # kh = k * self.mesh_spacing
        U2 = torch.prod(
            torch.sinc(kh / (2 * torch.pi)),
            dim=-1,
        ) ** self.interpolation_nodes
        return torch.stack(
            [
                torch.roll(U2, shifts=tuple(direction), dims=(0, 1, 2))
                for direction in self._directions
            ],
            dim=-1,
        )

        # return (
        #     torch.prod(
        #         torch.sinc(k * (self.mesh_spacing / 2 / torch.pi)),
        #         dim=-1,
        #     )
        #     ** self.interpolation_nodes
        # )

    def _reference_force(self, k: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd, 3)"""

        k_sq = torch.linalg.norm(k, dim=-1) ** 2
        R = k * self.lr_from_k_sq(k_sq)[..., torch.newaxis]
        return torch.stack(
            [
                torch.roll(R, shifts=tuple(direction), dims=(0, 1, 2))
                for direction in self._directions
            ],
            dim=-2,
        )

        # return k * self.lr_from_k_sq(k_sq)[..., torch.newaxis]
