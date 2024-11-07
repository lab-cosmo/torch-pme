from functools import cached_property
from typing import Optional

import torch
from torch.nn.functional import pad
from .coulomb import CoulombPotential

COEF = [
    [1],
    [4 / 3, -1 / 3],
    [3 / 2, -3 / 5, 1 / 10],
    [8 / 5, -4 / 5, 8 / 35, -1 / 35],
    [5 / 3, -20 / 21, 5 / 14, -5 / 63, 1 / 126],
    [12 / 7, -15 / 14, 10 / 21, -1 / 7, 2 / 77, -1 / 465],
]


class P3MCoulombPotential(CoulombPotential):
    r"""Coulomb potential for the P3M method."""

    def __init__(
        self,
        interpolation_nodes: int,
        mesh_spacing: torch.Tensor,
        cell: torch.Tensor,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        mode: int = 0,
        diff_order: int = 2,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)
        self.interpolation_nodes = interpolation_nodes
        self.mesh_spacing = mesh_spacing.reshape(1, 1, 1, 3)
        self.cell = cell
        self.mode = mode
        self.diff_order = torch.tensor(diff_order, dtype=torch.double, device=cell.device)
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
        """For km vectors calculation, deprecated for now, might be useful in the future"""
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
        """For km vectors calculation, partly deprecated for now since it's not used"""

        # For now, just return zero vector
        return torch.tensor([[0, 0, 0]]).int().numpy()
    
        # May be useful in the future
        # m = torch.tensor([-2, -1, 0, 1, 2], dtype=self.cell.dtype).# unsqueeze(-1)
        # m = (
        #     pad(input=m, pad=(0, 2))[:, None, None]
        #     + pad(input=m, pad=(1, 1))[None, :, None]
        #     + pad(input=m, pad=(2, 0))[None, None, 2:]  # no -z direction vector
        # ).reshape(-1, 3)
        # return m[torch.linalg.norm(m, dim=1) <= 2].int().numpy()

    @torch.jit.export
    def kernel_from_k(self, k: torch.Tensor) -> torch.Tensor:
        """
        Compatibility function with the interface of :py:class:`KSpaceKernel`, so that
        potentials can be used as kernels for :py:class:`KSpaceFilter`.
        """
        result = self.lr_from_k(k)
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
        if self.mode == 0:
            D = torch.ones(k.shape, dtype=k.dtype, device=k.device)
            D2 = torch.ones(k.shape[:3], dtype=k.dtype, device=k.device)
        else:
            D = self._diff_operator(kh)
            D2 = torch.linalg.norm(D, dim=-1) ** 2

        U2 = self._charge_assignment(kh) ** 2
        R = self._reference_force(k)
        numerator = torch.sum(
            (self.k_prime(k) @ D.unsqueeze(-1)).squeeze(-1) ** self.mode * U2 * R, dim=-1
        )
        denominator = D2**self.mode * torch.sum(U2, dim=-1) ** 2

        return torch.where(
            denominator == 0,
            0.0,
            numerator / denominator,
        )

    def _diff_operator(self, kh: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, 3)"""

        temp = torch.zeros(kh.shape, dtype=kh.dtype, device=kh.device)
        for i, coef in enumerate(COEF[self.diff_order - 1]):
            temp += (coef / (i + 1)) * torch.sin(kh * (i + 1))
        return temp / (self.mesh_spacing)

    def _charge_assignment(self, kh: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd)"""
        U2 = (
            torch.prod(
                torch.sinc(kh / (2 * torch.pi)),
                dim=-1,
            )
            ** self.interpolation_nodes
        )
        return torch.stack(
            [
                torch.roll(U2, shifts=tuple(direction), dims=(0, 1, 2))
                for direction in self._directions
            ],
            dim=-1,
        )

    def _reference_force(self, k: torch.Tensor) -> torch.Tensor:
        """
        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd, 3)"""

        k_sq = torch.linalg.norm(k, dim=-1) ** 2
        R = self.lr_from_k_sq(k_sq)
        return torch.stack(
            [
                torch.roll(R, shifts=tuple(direction), dims=(0, 1, 2))
                for direction in self._directions
            ],
            dim=-1,
        )

    def k_prime(self, k: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.roll(k, shifts=tuple(direction), dims=(0, 1, 2))
                for direction in self._directions
            ],
            dim=-2,
        )
