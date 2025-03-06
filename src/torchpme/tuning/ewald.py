import math
from typing import Any
from warnings import warn

import torch

from ..calculators import EwaldCalculator
from .tuner import GridSearchTuner, TuningErrorBounds


def tune_ewald(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    full_neighbor_list: bool = False,
    prefactor: float = 1.0,
    exponent: int = 1,
    ns_lo: int = 1,
    ns_hi: int = 14,
    accuracy: float = 1e-3,
) -> tuple[float, dict[str, Any], float]:
    r"""
    Find the optimal parameters for :class:`torchpme.EwaldCalculator`.

    .. note::

        The :func:`torchpme.tuning.ewald.EwaldErrorBounds.forward` method takes floats
        as the input, in order to be in consistency with the rest of the package --
        these parameters are always ``float`` but not ``torch.Tensor``. This design,
        however, prevents the utilization of ``torch.autograd`` and other ``torch``
        features. To take advantage of these features, one can use the
        :func:`torchpme.tuning.ewald.EwaldErrorBounds.err_rspace` and
        :func:`torchpme.tuning.ewald.EwaldErrorBounds.err_kspace`, which takes
        ``torch.Tensor`` as parameters.

    :param charges: torch.tensor of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
    :param cutoff: float, cutoff distance for the neighborlist
    :param exponent: :math:`p` in :math:`1/r^p` potentials, currently only :math:`p=1`
        is supported
    :param neighbor_indices: torch.tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.tensor with the pair distances of the neighbors for
        which the potential should be computed in real space.
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
    :param ns_lo: Minimum number of spatial resolution along each axis
    :param ns_hi: Maximum number of spatial resolution along each axis
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.

    :return: Tuple containing a float of the optimal smearing for the :class:
        `CoulombPotential`, and a dictionary with the parameters for
        :class:`EwaldCalculator`, and the timing of this set of parameters.

    Example
    -------
    >>> import torch
    >>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]])
    >>> charges = torch.tensor([[1.0], [-1.0]])
    >>> cell = torch.eye(3)
    >>> neighbor_distances = torch.tensor(
    ...     [0.9381, 0.9381, 0.8246, 0.9381, 0.8246, 0.8246, 0.6928],
    ... )
    >>> neighbor_indices = torch.tensor(
    ...     [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    ... )
    >>> smearing, parameter, timing = tune_ewald(
    ...     charges,
    ...     cell,
    ...     positions,
    ...     cutoff=1.0,
    ...     neighbor_distances=neighbor_distances,
    ...     neighbor_indices=neighbor_indices,
    ...     accuracy=1e-1,
    ... )

    """
    # if cell is 0 `min_dimension` will be zero as well; we raise a propper error later
    min_dimension = float(torch.min(torch.linalg.norm(cell, dim=1)))
    params = [{"lr_wavelength": min_dimension / ns} for ns in range(ns_lo, ns_hi + 1)]

    tuner = GridSearchTuner(
        charges=charges,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        exponent=exponent,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        full_neighbor_list=full_neighbor_list,
        prefactor=prefactor,
        calculator=EwaldCalculator,
        error_bounds=EwaldErrorBounds(charges=charges, cell=cell, positions=positions),
        params=params,
    )
    smearing = tuner.estimate_smearing(accuracy)
    errs, timings = tuner.tune(accuracy)

    # There are multiple errors below the accuracy, return the one with the shortest
    # calculation time. The timing of those parameters leading to an higher error than
    # the accuracy are set to infinity
    if any(err < accuracy for err in errs):
        return smearing, params[timings.index(min(timings))], min(timings)
    # No parameter meets the requirement, return the one with the smallest error
    warn(
        f"No parameter meets the accuracy requirement.\n"
        f"Returning the parameter with the smallest error, which is {min(errs)}.\n",
        stacklevel=1,
    )
    return smearing, params[errs.index(min(errs))], timings[errs.index(min(errs))]


class EwaldErrorBounds(TuningErrorBounds):
    r"""
    Error bounds for :class:`torchpme.calculators.ewald.EwaldCalculator`.

        .. math::
            \text{Error}_{\text{total}} = \sqrt{\text{Error}_{\text{real space}}^2 +
            \text{Error}_{\text{Fourier space}}^2

    :param charges: torch.tensor of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.

    Example
    -------
    >>> import torch
    >>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]])
    >>> charges = torch.tensor([[1.0], [-1.0]])
    >>> cell = torch.eye(3)
    >>> error_bounds = EwaldErrorBounds(charges, cell, positions)
    >>> print(error_bounds(smearing=1.0, lr_wavelength=0.5, cutoff=4.4))
    tensor(8.4304e-05)

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
        return (
            self.prefac**0.5
            / smearing
            / torch.sqrt((2 * torch.pi) ** 2 * self.volume / (lr_wavelength) ** 0.5)
            * torch.exp(-((2 * torch.pi) ** 2) * smearing**2 / (lr_wavelength))
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
        smearing = torch.tensor(smearing)
        lr_wavelength = torch.tensor(lr_wavelength)
        cutoff = torch.tensor(cutoff)
        return torch.sqrt(
            self.err_kspace(smearing, lr_wavelength) ** 2
            + self.err_rspace(smearing, cutoff) ** 2
        )
