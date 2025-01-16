import math
from itertools import product
from typing import Optional

import torch

from ..calculators import PMECalculator
from .tuner import GridSearchTuner, TuningErrorBounds

TWO_PI = 2 * math.pi


def tune_pme(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    exponent: int = 1,
    neighbor_indices: Optional[torch.Tensor] = None,
    neighbor_distances: Optional[torch.Tensor] = None,
    nodes_lo: int = 3,
    nodes_hi: int = 7,
    mesh_lo: int = 2,
    mesh_hi: int = 7,
    accuracy: float = 1e-3,
):
    r"""
    Find the optimal parameters for :class:`torchpme.PMECalculator`.

    For the error formulas are given `elsewhere <https://doi.org/10.1063/1.470043>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    :param charges: torch.Tensor, atomic (pseudo-)charges
    :param cell: torch.Tensor, periodic supercell for the system
    :param positions: torch.Tensor, Cartesian coordinates of the particles within
        the supercell.
    :param cutoff: float, cutoff distance for the neighborlist
    :param exponent: :math:`p` in :math:`1/r^p` potentials, currently only :math:`p=1` is
        supported
    :param neighbor_indices: torch.Tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.Tensor with the pair distances of the neighbors
        for which the potential should be computed in real space.
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.

    :return: Tuple containing a float of the optimal smearing for the :class:
        `CoulombPotential`, a dictionary with the parameters for
        :class:`PMECalculator` and a float of the optimal cutoff value for the
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
    >>> smearing, parameter = tune_pme(
    ...     charges, cell, positions, cutoff=4.4, accuracy=1e-1
    ... )

    You can check the values of the parameters

    >>> print(smearing)
    1.7140874893066034

    >>> print(parameter)
    {'interpolation_nodes': 4, 'mesh_spacing': 0.6666666666666666}

    """
    min_dimension = float(torch.min(torch.linalg.norm(cell, dim=1)))
    params = [
        {
            "interpolation_nodes": interpolation_nodes,
            "mesh_spacing": 2 * min_dimension / (2**ns - 1),
        }
        for interpolation_nodes, ns in product(
            range(nodes_lo, nodes_hi + 1), range(mesh_lo, mesh_hi + 1)
        )
    ]

    tuner = GridSearchTuner(
        charges,
        cell,
        positions,
        cutoff,
        exponent=exponent,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        calculator=PMECalculator,
        error_bounds=PMEErrorBounds(charges=charges, cell=cell, positions=positions),
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

    def err_kspace(
        self,
        smearing: torch.Tensor,
        mesh_spacing: torch.Tensor,
        interpolation_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """
        The Fourier space error of PME.

        :param smearing: see :class:`torchpme.PMECalculator` for details
        :param mesh_spacing: see :class:`torchpme.PMECalculator` for details
        :param interpolation_nodes: see :class:`torchpme.PMECalculator` for details
        """
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

    def err_rspace(self, smearing: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """
        The real space error of PME.

        :param smearing: see :class:`torchpme.PMECalculator` for details
        :param cutoff: see :class:`torchpme.PMECalculator` for details
        """
        smearing = torch.as_tensor(smearing)
        cutoff = torch.as_tensor(cutoff)

        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def error(
        self,
        cutoff: float,
        smearing: float,
        mesh_spacing: float,
        interpolation_nodes: float,
    ) -> torch.Tensor:
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
