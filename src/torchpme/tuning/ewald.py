import math
from typing import Optional

import torch

from ..calculators import EwaldCalculator
from ..utils import _validate_parameters
from .tuner import GridSearchTuner, TuningErrorBounds

TWO_PI = 2 * math.pi


def tune_ewald(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    exponent: int = 1,
    neighbor_indices: Optional[torch.Tensor] = None,
    neighbor_distances: Optional[torch.Tensor] = None,
    ns_lo: int = 1,
    ns_hi: int = 14,
    accuracy: float = 1e-3,
) -> tuple[float, dict[str, float]]:
    r"""
    Find the optimal parameters for :class:`torchpme.EwaldCalculator`.

    The error formulas are given `online
    <https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/4/4d/Script_Longrange_Interactions.pdf>`_
    (now not available, need to be updated later). Note the difference notation between
    the parameters in the reference and ours:

    .. math::

        \alpha &= \left( \sqrt{2}\,\mathrm{smearing} \right)^{-1}

        K &= \frac{2 \pi}{\mathrm{lr\_wavelength}}

        r_c &= \mathrm{cutoff}

    :param charges: torch.Tensor, atomic (pseudo-)charges
    :param cell: torch.Tensor, periodic supercell for the system
    :param positions: torch.Tensor, Cartesian coordinates of the particles within the
        supercell.
    :param cutoff: float, cutoff distance for the neighborlist
    :param exponent: :math:`p` in :math:`1/r^p` potentials, currently only :math:`p=1`
        is supported
    :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.Tensor with the pair distances of the neighbors for
        which the potential should be computed in real space.
    :param ns_lo: Minimum number of spatial resolution along each axis
    :param ns_hi: Maximum number of spatial resolution along each axis
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.

    :return: Tuple containing a float of the optimal smearing for the :class:
        `CoulombPotential`, and a dictionary with the parameters for
        :class:`EwaldCalculator`.

    Example
    -------
    >>> import torch
    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)
    >>> smearing, parameter = tune_ewald(
    ...     charges, cell, positions, cutoff=4.4, accuracy=1e-1
    ... )

    """
    _validate_parameters(charges, cell, positions, exponent)
    min_dimension = float(torch.min(torch.linalg.norm(cell, dim=1)))
    params = [{"lr_wavelength": min_dimension / ns} for ns in range(ns_lo, ns_hi + 1)]
    tuner = GridSearchTuner(
        charges,
        cell,
        positions,
        cutoff,
        exponent=exponent,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        calculator=EwaldCalculator,
        error_bounds=EwaldErrorBounds(charges=charges, cell=cell, positions=positions),
        params=params,
    )
    smearing = tuner.estimate_smearing(accuracy)
    errs, timings = tuner.tune(accuracy)

    if any(err < accuracy for err in errs):
        # There are multiple errors below the accuracy, return the one with the shortest
        # calculation time. The timing of those parameters leading to an higher error
        # than the accuracy are set to infinity
        return smearing, params[timings.index(min(timings))]
    # No parameter meets the requirement, return the one with the smallest error
    return smearing, params[errs.index(min(errs))]


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

    Example
    -------
    >>> import torch
    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)
    >>> error_bounds = EwaldErrorBounds(charges, cell, positions)
    >>> print(error_bounds(smearing=1.0, lr_wavelength=0.5, cutoff=4.4))
    tensor(8.4304e-05, dtype=torch.float64)

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

    def err_kspace(
        self, smearing: torch.Tensor, lr_wavelength: torch.Tensor
    ) -> torch.Tensor:
        """
        The Fourier space error of Ewald.

        :param smearing: see :class:`torchpme.EwaldCalculator` for details
        :param lr_wavelength: see :class:`torchpme.EwaldCalculator` for details
        """
        smearing = torch.as_tensor(smearing)
        lr_wavelength = torch.as_tensor(lr_wavelength)
        return (
            self.prefac**0.5
            / smearing
            / torch.sqrt(TWO_PI**2 * self.volume / (lr_wavelength) ** 0.5)
            * torch.exp(-(TWO_PI**2) * smearing**2 / (lr_wavelength))
        )

    def err_rspace(self, smearing: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """
        The real space error of Ewald.

        :param smearing: see :class:`torchpme.EwaldCalculator` for details
        :param lr_wavelength: see :class:`torchpme.EwaldCalculator` for details
        """
        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def forward(
        self, smearing: float, lr_wavelength: float, cutoff: float
    ) -> torch.Tensor:
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
