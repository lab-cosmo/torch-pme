import math
from typing import Optional

import torch

from ..calculators import EwaldCalculator
from .tuner import GridSearchTuner

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
):
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

    :return: Tuple containing a float of the optimal smearing for the :class:
        `CoulombPotential`, a dictionary with the parameters for
        :class:`EwaldCalculator` and a float of the optimal cutoff value for the
        neighborlist computation.

    Example
    -------
    >>> import torch
    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)
    >>> smearing, parameter, cutoff = tune_ewald(
    ...     charges, cell, positions, cutoff=4.4, accuracy=1e-1
    ... )

    You can check the values of the parameters

    >>> print(smearing)
    1.7140874893066034

    >>> print(parameter)
    {'lr_wavelength': 0.25}

    >>> print(cutoff)
    4.4

    """
    params = [{"lr_wavelength": ns} for ns in range(ns_lo, ns_hi + 1)]
    tuner = GridSearchTuner(
        charges,
        cell,
        positions,
        cutoff,
        exponent=exponent,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        calculator=EwaldCalculator,
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
