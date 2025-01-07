import math
from typing import Optional

import numpy as np
import torch

from torchpme import P3MCalculator

from . import (
    TuningErrorBounds,
)
from .grid_search import GridSearchBase

# Coefficients for the P3M Fourier error,
# see Table II of http://dx.doi.org/10.1063/1.477415
A_COEF = [
    [None, 2 / 3, 1 / 50, 1 / 588, 1 / 4320, 1 / 23_232, 691 / 68_140_800, 1 / 345_600],
    [
        None,
        None,
        5 / 294,
        7 / 1440,
        3 / 1936,
        7601 / 13_628_160,
        13 / 57_600,
        3617 / 35_512_320,
    ],
    [
        None,
        None,
        None,
        21 / 3872,
        7601 / 2_271_360,
        143 / 69_120,
        47_021 / 35_512_320,
        745_739 / 838_397_952,
    ],
    [
        None,
        None,
        None,
        None,
        143 / 28_800,
        517_231 / 106_536_960,
        9_694_607 / 2_095_994_880,
        56_399_353 / 12_773_376_000,
    ],
    [
        None,
        None,
        None,
        None,
        None,
        106_640_677 / 11_737_571_328,
        733_191_589 / 59_609_088_000,
        25_091_609 / 1_560_084_480,
    ],
    [
        None,
        None,
        None,
        None,
        None,
        None,
        326_190_917 / 11_700_633_600,
        1_755_948_832_039 / 36_229_939_200_000,
    ],
    [None, None, None, None, None, None, None, 4_887_769_399 / 37_838_389_248],
]


class P3MErrorBounds(TuningErrorBounds):
    r"""
    "
    Error bounds for :class:`torchpme.calculators.pme.P3MCalculator`.

    For the error formulas are given `here <https://doi.org/10.1063/1.477415>`_.
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
        self.cell = cell
        self.positions = positions

    def err_kspace(self, smearing, mesh_spacing, interpolation_nodes):
        actual_spacing = self.cell_dimensions / (
            2 * self.cell_dimensions / mesh_spacing + 1
        )
        h = torch.prod(actual_spacing) ** (1 / 3)

        return (
            self.prefac
            / self.volume ** (2 / 3)
            * (h * (1 / 2**0.5 / smearing)) ** interpolation_nodes
            * torch.sqrt(
                (1 / 2**0.5 / smearing)
                * self.volume ** (1 / 3)
                * math.sqrt(2 * torch.pi)
                * sum(
                    A_COEF[m][interpolation_nodes]
                    * (h * (1 / 2**0.5 / smearing)) ** (2 * m)
                    for m in range(interpolation_nodes)
                )
            )
        )

    def err_rspace(self, smearing, cutoff):
        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def forward(self, smearing, mesh_spacing, cutoff, interpolation_nodes):
        r"""
        Calculate the error bound of P3M.

        :param smearing: see :class:`torchpme.P3MCalculator` for details
        :param mesh_spacing: see :class:`torchpme.P3MCalculator` for details
        :param cutoff: see :class:`torchpme.P3MCalculator` for details
        :param interpolation_nodes: The number ``n`` of nodes used in the interpolation
            per coordinate axis. The total number of interpolation nodes in 3D will be
            ``n^3``. In general, for ``n`` nodes, the interpolation will be performed by
            piecewise polynomials of degree ``n`` (e.g. ``n = 3`` for cubic
            interpolation). Only the values ``1, 2, 3, 4, 5`` are supported.
        """
        smearing = torch.as_tensor(smearing)
        mesh_spacing = torch.as_tensor(mesh_spacing)
        cutoff = torch.as_tensor(cutoff)
        interpolation_nodes = torch.as_tensor(interpolation_nodes)
        return torch.sqrt(
            self.err_kspace(smearing, mesh_spacing, interpolation_nodes) ** 2
            + self.err_rspace(smearing, cutoff) ** 2
        )


class P3MTuner(GridSearchBase):
    """
    Class for finding the optimal parameters for P3MCalculator using a grid search.

    For details of the parameters see :class:`torchpme.utils.tuning.GridSearchBase`.
    """

    ErrorBounds = P3MErrorBounds
    CalculatorClass = P3MCalculator
    TemplateGridSearchParams = {
        "interpolation_nodes": [2, 3, 4, 5],
        "mesh_spacing": 1
        / ((np.exp2(np.arange(2, 8)) - 1) / 2),  # will be converted into a list later
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
        self.GridSearchParams["mesh_spacing"] = self.GridSearchParams["mesh_spacing"].tolist()


def tune_p3m(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    exponent: int = 1,
    neighbor_indices: Optional[torch.Tensor] = None,
    neighbor_distances: Optional[torch.Tensor] = None,
    accuracy: float = 1e-3,
):
    r"""
    Find the optimal parameters for :class:`torchpme.calculators.pme.PMECalculator`.

    For the error formulas are given `here <https://doi.org/10.1063/1.477415>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    :param charges: torch.Tensor, atomic (pseudo-)charges
    :param cell: torch.Tensor, periodic supercell for the system
    :param positions: torch.Tensor, Cartesian coordinates of the particles within
        the supercell.
    :param cutoff: float, cutoff distance for the neighborlist
    :param exponent :math:`p` in :math:`1/r^p` potentials, currently only :math:`p=1` is
        supported
    :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.Tensor with the pair distances of the neighbors
        for which the potential should be computed in real space.
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.

    :return: Tuple containing a float of the optimal smearing for the :py:class:
        `CoulombPotential`, a dictionary with the parameters for
        :py:class:`PMECalculator` and a float of the optimal cutoff value for the
        neighborlist computation.

    Example
    -------
    >>> import torch

    To allow reproducibility, we set the seed to a fixed value

    >>> _ = torch.manual_seed(0)
    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)
    >>> smearing, parameter, cutoff = tune_p3m(
    ...     charges, cell, positions, cutoff=4.4, accuracy=1e-1
    ... )

    You can check the values of the parameters

    >>> print(smearing)
    1.7140874893066034

    >>> print(parameter)
    {'interpolation_nodes': 2, 'mesh_spacing': 0.2857142857142857}

    >>> print(cutoff)
    4.4

    """
    return P3MTuner(
        charges, cell, positions, cutoff, exponent, neighbor_indices, neighbor_distances
    ).tune(accuracy)
