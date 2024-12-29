import math
from typing import Optional

import numpy as np
import torch

from torchpme import PMECalculator

from . import (
    TuningErrorBounds,
)
from .grid_search import GridSearchBase


class PMEErrorBounds(TuningErrorBounds):
    r"""
    Error bounds for :class:`torchpme.PMECalculator`.
    For the error formulas are given `elsewhere <https://doi.org/10.1063/1.470043>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    """

    def __init__(
        self, charges: torch.Tensor, cell: torch.Tensor, positions: torch.Tensor
    ):
        super().__init__(charges, cell, positions)

        self.volume = torch.abs(torch.det(cell))
        self.sum_squared_charges = (charges**2).sum()
        self.prefac = 2 * self.sum_squared_charges / math.sqrt(len(positions))
        self.cell_dimensions = torch.linalg.norm(cell, dim=1)

    def err_kspace(self, smearing, mesh_spacing, interpolation_nodes):
        actual_spacing = self.cell_dimensions / (
            2 * self.cell_dimensions / mesh_spacing + 1
        )
        h = torch.prod(actual_spacing) ** (1 / 3)
        i_n_factorial = torch.exp(torch.lgamma(interpolation_nodes + 1))
        RMS_phi = [None, None, 0.246, 0.404, 0.950, 2.51, 8.42]

        return (
            self.prefac
            * torch.pi**0.25
            * (6 * (1 / 2**0.5 / smearing) / (2 * interpolation_nodes + 1)) ** 0.5
            / self.volume ** (2 / 3)
            * (2**0.5 / smearing * h) ** interpolation_nodes
            / i_n_factorial
            * torch.exp(
                interpolation_nodes * (torch.log(interpolation_nodes / 2) - 1) / 2
            )
            * RMS_phi[interpolation_nodes - 1]
        )

    def err_rspace(self, smearing, cutoff):
        smearing = torch.as_tensor(smearing)
        cutoff = torch.as_tensor(cutoff)

        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def error(self, cutoff, smearing, mesh_spacing, interpolation_nodes):
        r"""
        Calculate the error bound of PME.

        :param smearing: if its value is given, it will not be tuned, see
            :class:`torchpme.PMECalculator` for details
        :param mesh_spacing: if its value is given, it will not be tuned, see
            :class:`torchpme.PMECalculator` for details
        :param cutoff: if its value is given, it will not be tuned, see
            :class:`torchpme.PMECalculator` for details
        :param interpolation_nodes: The number ``n`` of nodes used in the interpolation
            per coordinate axis. The total number of interpolation nodes in 3D will be
            ``n^3``. In general, for ``n`` nodes, the interpolation will be performed by
            piecewise polynomials of degree ``n - 1`` (e.g. ``n = 4`` for cubic
            interpolation). Only the values ``3, 4, 5, 6, 7`` are supported.
        """
        smearing = torch.as_tensor(smearing)
        mesh_spacing = torch.as_tensor(mesh_spacing)
        cutoff = torch.as_tensor(cutoff)
        interpolation_nodes = torch.as_tensor(interpolation_nodes)
        return torch.sqrt(
            self.err_rspace(smearing, cutoff) ** 2
            + self.err_kspace(smearing, mesh_spacing, interpolation_nodes) ** 2
        )


class PMETuner(GridSearchBase):
    """
    Class for finding the optimal parameters for PMECalculator using a grid search.

    For details of the parameters see :class:`torchpme.utils.tuning.GridSearchBase`.
    """

    ErrorBounds = PMEErrorBounds
    CalculatorClass = PMECalculator
    GridSearchParams = {
        "interpolation_nodes": [3, 4, 5, 6, 7],
        "mesh_spacing": 1 / ((np.exp2(np.arange(2, 8)) - 1) / 2),
    }

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        cutoff: float,
        exponent: int = 1,
        neighbor_indices: Optional[torch.Tensor] = None,
        neighbor_distances: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            charges,
            cell,
            positions,
            cutoff,
            exponent,
            neighbor_indices,
            neighbor_distances,
        )
        self.GridSearchParams["mesh_spacing"] *= float(torch.min(self._cell_dimensions))
