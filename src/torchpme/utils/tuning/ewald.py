import math
from typing import Optional

import numpy as np
import torch

from ...calculators import EwaldCalculator
from . import (
    TuningErrorBounds,
)
from .grid_search import GridSearchBase

TWO_PI = 2 * math.pi


class EwaldErrorBounds(TuningErrorBounds):
    r"""
    Error bounds for :class:`torchpme.calculators.ewald.EwaldCalculator`.

    The error formulas are given `online
    <https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/4/4d/Script_Longrange_Interactions.pdf>`_
    (now not available, need to be updated later). Note the difference notation between
    the parameters in the reference and ours:

    .. math::

        \alpha &= \left( \sqrt{2}\,\mathrm{smearing} \right)^{-1}

        K &= \frac{2 \pi}{\mathrm{lr\_wavelength}}

        r_c &= \mathrm{cutoff}

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ):
        super().__init__(charges, cell, positions)

        self.volume = torch.abs(torch.det(cell))
        self.sum_squared_charges = (charges**2).sum()
        self.prefac = 2 * self.sum_squared_charges / math.sqrt(len(positions))
        self.cell = cell
        self.positions = positions

    def err_kspace(self, smearing, lr_wavelength):
        return (
            self.prefac**0.5
            / smearing
            / torch.sqrt(TWO_PI**2 * self.volume / (lr_wavelength) ** 0.5)
            * torch.exp(-(TWO_PI**2) * smearing**2 / (lr_wavelength))
        )

    def err_rspace(self, smearing, cutoff):
        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def forward(self, smearing, lr_wavelength, cutoff):
        r"""
        Calculate the error bound of Ewald.
        :param smearing: see :class:`torchpme.EwaldCalculator` for details
        :param lr_wavelength: see :class:`torchpme.EwaldCalculator` for details
        :param cutoff: see :class:`torchpme.EwaldCalculator` for details
        """
        smearing = torch.as_tensor(smearing)
        lr_wavelength = torch.as_tensor(lr_wavelength)
        cutoff = torch.as_tensor(cutoff)
        return torch.sqrt(
            self.err_kspace(smearing, lr_wavelength) ** 2
            + self.err_rspace(smearing, cutoff) ** 2
        )


class EwaldTuner(GridSearchBase):
    """
    Class for finding the optimal parameters for EwaldCalculator using a grid search.

    For details of the parameters see :class:`torchpme.utils.tuning.GridSearchBase`.
    """

    ErrorBounds = EwaldErrorBounds
    CalculatorClass = EwaldCalculator
    GridSearchParams = {"lr_wavelength": 1 / np.arange(1, 15)}

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
        self.GridSearchParams["lr_wavelength"] *= float(
            torch.min(self._cell_dimensions)
        )

    __doc__ = GridSearchBase.__doc__
