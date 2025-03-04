import math
from itertools import product
from typing import Any
from warnings import warn

import torch

from ..calculators import PMECalculator
from .tuner import GridSearchTuner, TuningErrorBounds


def tune_pme(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    full_neighbor_list: bool = False,
    prefactor: float = 1.0,
    exponent: int = 1,
    nodes_lo: int = 3,
    nodes_hi: int = 7,
    mesh_lo: int = 2,
    mesh_hi: int = 7,
    accuracy: float = 1e-3,
) -> tuple[float, dict[str, Any], float]:
    r"""
    Find the optimal parameters for :class:`torchpme.PMECalculator`.

    For the error formulas are given `elsewhere <https://doi.org/10.1063/1.470043>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    :param charges: torch.tensor of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
    :param cutoff: float, cutoff distance for the neighborlist
    :param neighbor_indices: torch.tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.tensor with the pair distances of the neighbors for
        which the potential should be computed in real space.
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
    :param exponent: :math:`p` in :math:`1/r^p` potentials, currently only :math:`p=1`
        is supported
    :param nodes_lo: Minimum number of interpolation nodes
    :param nodes_hi: Maximum number of interpolation nodes
    :param mesh_lo: Controls the minimum number of mesh points along the shortest axis,
        :math:`2^{mesh_lo}`
    :param mesh_hi: Controls the maximum number of mesh points along the shortest axis,
        :math:`2^{mesh_hi}`
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.

    :return: Tuple containing a float of the optimal smearing for the :class:
        `CoulombPotential`, a dictionary with the parameters for :class:`PMECalculator`
        and a float of the optimal cutoff value for the neighborlist computation, and
        the timing of this set of parameters.

    Example
    -------
    >>> import torch

    To allow reproducibility, we set the seed to a fixed value

    >>> _ = torch.manual_seed(0)
    >>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    >>> charges = torch.tensor([[1.0], [-1.0]])
    >>> cell = torch.eye(3)
    >>> neighbor_distances = torch.tensor(
    ...     [0.9381, 0.9381, 0.8246, 0.9381, 0.8246, 0.8246, 0.6928],
    ... )
    >>> neighbor_indices = torch.tensor(
    ...     [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    ... )
    >>> smearing, parameter, timing = tune_pme(
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
        charges=charges,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        exponent=exponent,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        full_neighbor_list=full_neighbor_list,
        prefactor=prefactor,
        calculator=PMECalculator,
        error_bounds=PMEErrorBounds(charges=charges, cell=cell, positions=positions),
        params=params,
    )
    smearing = tuner.estimate_smearing(accuracy)
    errs, timings = tuner.tune(accuracy)

    # There are multiple errors below the accuracy, return the one with the shortest
    # calculation time. The timing of those parameters leading to an higher error
    # than the accuracy are set to infinity
    if any(err < accuracy for err in errs):
        return smearing, params[timings.index(min(timings))], min(timings)
    # No parameter meets the requirement, return the one with the smallest error, and
    # throw a warning
    warn(
        f"No parameter meets the accuracy requirement.\n"
        f"Returning the parameter with the smallest error, which is {min(errs)}.\n",
        stacklevel=1,
    )
    return smearing, params[errs.index(min(errs))], timings[errs.index(min(errs))]


class PMEErrorBounds(TuningErrorBounds):
    r"""
    Error bounds for :class:`torchpme.PMECalculator`.

    .. note::

        The :func:`torchpme.tuning.pme.PMEErrorBounds.forward` method takes floats as
        the input, in order to be in consistency with the rest of the package -- these
        parameters are always ``float`` but not ``torch.Tensor``. This design, however,
        prevents the utilization of ``torch.autograd`` and other ``torch`` features. To
        take advantage of these features, one can use the
        :func:`torchpme.tuning.pme.PMEErrorBounds.err_rspace` and
        :func:`torchpme.tuning.pme.PMEErrorBounds.err_kspace`, which takes
        ``torch.Tensor`` as parameters.

    :param charges: torch.tensor of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.

    Example
    -------
    >>> import torch
    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)
    >>> error_bounds = PMEErrorBounds(charges, cell, positions)
    >>> print(
    ...     error_bounds(
    ...         smearing=1.0, mesh_spacing=0.5, cutoff=4.4, interpolation_nodes=3
    ...     )
    ... )
    tensor(0.0011, dtype=torch.float64)

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

        .. math::
            \text{Error}_{\text{total}} = \sqrt{\text{Error}_{\text{real space}}^2 +
            \text{Error}_{\text{Fourier space}}^2

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
        smearing = torch.tensor(smearing)
        mesh_spacing = torch.tensor(mesh_spacing)
        cutoff = torch.tensor(cutoff)
        interpolation_nodes = torch.tensor(interpolation_nodes)
        return torch.sqrt(
            self.err_rspace(smearing, cutoff) ** 2
            + self.err_kspace(smearing, mesh_spacing, interpolation_nodes) ** 2
        )
