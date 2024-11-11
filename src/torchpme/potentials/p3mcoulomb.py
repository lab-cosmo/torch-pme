from functools import cached_property
from typing import Optional

import torch
from torch.nn.functional import pad

from ..lib.kvectors import get_ns_mesh
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
    r"""Coulomb potential for the P3M method.

    :param smearing: float or torch.Tensor containing the parameter often called "sigma"
        in publications, which determines the length-scale at which the short-range and
        long-range parts of the naive :math:`1/r` potential are separated. The smearing
        parameter corresponds to the "width" of a Gaussian smearing of the particle
        density.
    :param exclusion_radius: A length scale that defines a *local environment* within
        which the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param mode: int, 0 for the electrostatic potential, 1 for the electrostatic energy,
        2 for the dipolar torques, and 3 for the dipolar forces. For more details, see
        eq.30 of `this paper<https://doi.org/10.1063/1.3000389>`_
    :param diff_order: int, the order of the approximation of the difference operator.
        Higher order is more accurate, but also more expensive. For more details, see
        Appendix C of `that paper<http://dx.doi.org/10.1063/1.477414>`_. The values ``1,
        2, 3, 4, 5, 6`` are supported.
    :param dtype: type used for the internal buffers and parameters
    :param device: device used for the internal buffers and parameters"""

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        mode: int = 0,
        diff_order: int = 2,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)
        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"`mode` should be one of [0, 1, 2, 3], but got {mode}")
        self.mode = mode

        if diff_order not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"`diff_order` should be one of [1, 2, 3, 4, 5, 6], but got {diff_order}"
            )
        self.diff_order = diff_order

        # Dummy variables for initialization
        self._update_cell(torch.eye(3))
        self._update_potential(1.0, 1)

    def _update_cell(self, cell: torch.Tensor):
        self._cell = cell
        if cell.is_cuda:
            # use function that does not synchronize with the CPU
            inverse_cell = torch.linalg.inv_ex(cell)[0]
        else:
            inverse_cell = torch.linalg.inv(cell)
        self._reciprocal_cell = 2 * torch.pi * inverse_cell.T

    def _update_potential(self, mesh_spacing: float, interpolation_nodes: int):
        self._crude_mesh_spacing = mesh_spacing
        self.interpolation_nodes = interpolation_nodes

    @property
    def mesh_spacing(self) -> torch.Tensor:
        cell_dimensions = torch.linalg.norm(self._cell, dim=1)
        return (
            cell_dimensions / get_ns_mesh(self._cell, self._crude_mesh_spacing)
        ).reshape(1, 1, 1, 3)

    @cached_property
    def _k_vectors(self) -> torch.Tensor:
        """For km vectors calculation, deprecated for now, might be useful in the future"""
        m = torch.tensor([-2, -1, 0, 1, 2], dtype=self._cell.dtype).unsqueeze(-1)
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
            (self.k_prime(k) @ D.unsqueeze(-1)).squeeze(-1) ** self.mode * U2 * R,
            dim=-1,
        )
        denominator = D2**self.mode * torch.sum(U2, dim=-1) ** 2

        return torch.where(
            denominator == 0,
            0.0,
            numerator / denominator,
        )

    def _diff_operator(self, kh: torch.Tensor) -> torch.Tensor:
        """From shape (nx, ny, nz, 3) to shape (nx, ny, nz, 3)"""
        temp = torch.zeros(kh.shape, dtype=kh.dtype, device=kh.device)
        for i, coef in enumerate(COEF[self.diff_order - 1]):
            temp += (coef / (i + 1)) * torch.sin(kh * (i + 1))
        return temp / (self.mesh_spacing)

    def _charge_assignment(self, kh: torch.Tensor) -> torch.Tensor:
        """From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd)"""
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
        """From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd, 3)"""
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
