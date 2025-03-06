import math
from itertools import product
from typing import Any
from warnings import warn

import torch

from ..calculators import P3MCalculator
from .tuner import GridSearchTuner, TuningErrorBounds

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


def tune_p3m(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    full_neighbor_list: bool = False,
    prefactor: float = 1.0,
    exponent: int = 1,
    nodes_lo: int = 2,
    nodes_hi: int = 5,
    mesh_lo: int = 2,
    mesh_hi: int = 7,
    accuracy: float = 1e-3,
) -> tuple[float, dict[str, Any], float]:
    r"""
    Find the optimal parameters for :class:`torchpme.calculators.pme.PMECalculator`.

    For the error formulas are given `here <https://doi.org/10.1063/1.477415>`_. Note
    the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    :param charges: torch.tensor of shape ``(len(positions, 1))`` containing the atomic
        (pseudo-)charges
    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param positions: torch.tensor of shape ``(N, 3)`` containing the Cartesian
        coordinates of the ``N`` particles within the supercell.
    :param neighbor_indices: torch.tensor with the ``i,j`` indices of neighbors for
        which the potential should be computed in real space.
    :param neighbor_distances: torch.tensor with the pair distances of the neighbors for
        which the potential should be computed in real space.
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
    :param cutoff: float, cutoff distance for the neighborlist supported
    :param exponent: :math:`p` in :math:`1/r^p` potentials, currently only :math:`p=1`
        is
    :param nodes_lo: Minimum number of interpolation nodes
    :param nodes_hi: Maximum number of interpolation nodes
    :param mesh_lo: Controls the minimum number of mesh points along the shortest axis,
        :math:`2^{mesh_lo}`
    :param mesh_hi: Controls the maximum number of mesh points along the shortest axis,
        :math:`2^{mesh_hi}`
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.

    :return: Tuple containing a float of the optimal smearing for the :py:class:
        `CoulombPotential`, a dictionary with the parameters for
        :py:class:`P3MCalculator` and a float of the optimal cutoff value for the
        neighborlist computation, and the timing of this set of parameters.

    Example
    -------
    >>> import torch

    To allow reproducibility, we set the seed to a fixed value

    >>> _ = torch.manual_seed(0)
    >>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]])
    >>> charges = torch.tensor([[1.0], [-1.0]])
    >>> cell = torch.eye(3)
    >>> neighbor_distances = torch.tensor(
    ...     [0.9381, 0.9381, 0.8246, 0.9381, 0.8246, 0.8246, 0.6928],
    ... )
    >>> neighbor_indices = torch.tensor(
    ...     [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    ... )
    >>> smearing, parameter, timing = tune_p3m(
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
        calculator=P3MCalculator,
        error_bounds=P3MErrorBounds(charges=charges, cell=cell, positions=positions),
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


class P3MErrorBounds(TuningErrorBounds):
    r"""
    " Error bounds for :class:`torchpme.calculators.pme.P3MCalculator`.

    .. note::

        The :func:`torchpme.tuning.p3m.P3MErrorBounds.forward` method takes floats as
        the input, in order to be in consistency with the rest of the package -- these
        parameters are always ``float`` but not ``torch.Tensor``. This design, however,
        prevents the utilization of ``torch.autograd`` and other ``torch`` features. To
        take advantage of these features, one can use the
        :func:`torchpme.tuning.p3m.P3MErrorBounds.err_rspace` and
        :func:`torchpme.tuning.p3m.P3MErrorBounds.err_kspace`, which takes
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
    >>> error_bounds = P3MErrorBounds(charges, cell, positions)
    >>> print(
    ...     error_bounds(
    ...         smearing=1.0, mesh_spacing=0.5, cutoff=4.4, interpolation_nodes=3
    ...     )
    ... )
    tensor(0.0005, dtype=torch.float64)

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

    def err_kspace(
        self,
        smearing: torch.Tensor,
        mesh_spacing: torch.Tensor,
        interpolation_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """
        The Fourier space error of P3M.

        :param smearing: see :class:`torchpme.P3MCalculator` for details
        :param mesh_spacing: see :class:`torchpme.P3MCalculator` for details
        :param interpolation_nodes: see :class:`torchpme.P3MCalculator` for details
        """
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

    def err_rspace(self, smearing: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """
        The real space error of P3M.

        :param smearing: see :class:`torchpme.P3MCalculator` for details
        :param cutoff: see :class:`torchpme.P3MCalculator` for details
        """
        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def forward(
        self,
        smearing: float,
        mesh_spacing: float,
        cutoff: float,
        interpolation_nodes: int,
    ) -> torch.Tensor:
        r"""
        Calculate the error bound of P3M.

        .. math::
            \text{Error}_{\text{total}} = \sqrt{\text{Error}_{\text{real space}}^2 +
            \text{Error}_{\text{Fourier space}}^2

        :param smearing: see :class:`torchpme.P3MCalculator` for details
        :param mesh_spacing: see :class:`torchpme.P3MCalculator` for details
        :param cutoff: see :class:`torchpme.P3MCalculator` for details
        :param interpolation_nodes: see :class:`torchpme.P3MCalculator` for details
        """
        smearing = torch.tensor(smearing)
        mesh_spacing = torch.tensor(mesh_spacing)
        cutoff = torch.tensor(cutoff)
        interpolation_nodes = torch.tensor(interpolation_nodes)
        return torch.sqrt(
            self.err_kspace(smearing, mesh_spacing, interpolation_nodes) ** 2
            + self.err_rspace(smearing, cutoff) ** 2
        )
