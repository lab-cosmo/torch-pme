import math
from typing import Optional

import torch

from ...lib import get_ns_mesh
from . import (
    _estimate_smearing_cutoff,
    _optimize_parameters,
    _validate_parameters,
)

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
    sum_squared_charges: float,
    cell: torch.Tensor,
    positions: torch.Tensor,
    smearing: Optional[float] = None,
    mesh_spacing: Optional[float] = None,
    cutoff: Optional[float] = None,
    interpolation_nodes: int = 4,
    exponent: int = 1,
    accuracy: float = 1e-3,
    max_steps: int = 50000,
    learning_rate: float = 5e-3,
) -> tuple[float, dict[str, float], float]:
    r"""
    Find the optimal parameters for :class:`torchpme.calculators.pme.PMECalculator`.

    For the error formulas are given `here <https://doi.org/10.1063/1.477415>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    .. hint::

        Tuning uses an initial guess for the optimization, which can be applied by
        setting ``max_steps = 0``. This can be useful if fast tuning is required. These
        values typically result in accuracies around :math:`10^{-2}`.

    :param sum_squared_charges: accumulated squared charges, must be positive
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param interpolation_nodes: The number ``n`` of nodes used in the interpolation per
        coordinate axis. The total number of interpolation nodes in 3D will be ``n^3``.
        In general, for ``n`` nodes, the interpolation will be performed by piecewise
        polynomials of degree ``n`` (e.g. ``n = 3`` for cubic interpolation). Only
        the values ``1, 2, 3, 4, 5`` are supported.
    :param exponent: exponent :math:`p` in :math:`1/r^p` potentials
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.
    :param max_steps: maximum number of gradient descent steps
    :param learning_rate: learning rate for gradient descent
    :param verbose: whether to print the progress of gradient descent

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
    ...     torch.sum(charges**2, dim=0), cell, positions, accuracy=1e-1
    ... )

    You can check the values of the parameters

    >>> print(smearing)
    0.5084014996119913

    >>> print(parameter)
    {'mesh_spacing': 0.546694745583215, 'interpolation_nodes': 4}

    >>> print(cutoff)
    2.6863848597963442

    """
    _validate_parameters(sum_squared_charges, cell, positions, exponent, accuracy)

    smearing_opt, cutoff_opt = _estimate_smearing_cutoff(
        cell=cell,
        smearing=smearing,
        cutoff=cutoff,
        accuracy=accuracy,
    )
    # We choose only one mesh as initial guess
    if mesh_spacing is None:
        ns_mesh_opt = torch.tensor(
            [1, 1, 1],
            device=cell.device,
            dtype=cell.dtype,
            requires_grad=True,
        )
    else:
        ns_mesh_opt = get_ns_mesh(cell, mesh_spacing)

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    volume = torch.abs(cell.det())
    prefac = 2 * sum_squared_charges / math.sqrt(len(positions))

    interpolation_nodes = torch.tensor(interpolation_nodes, device=cell.device)

    def err_Fourier(smearing, ns_mesh):
        spacing = cell_dimensions / ns_mesh
        h = torch.prod(spacing) ** (1 / 3)

        return (
            prefac
            / volume ** (2 / 3)
            * (h * (1 / 2**0.5 / smearing)) ** interpolation_nodes
            * torch.sqrt(
                (1 / 2**0.5 / smearing)
                * volume ** (1 / 3)
                * math.sqrt(2 * torch.pi)
                * sum(
                    A_COEF[m][interpolation_nodes]
                    * (h * (1 / 2**0.5 / smearing)) ** (2 * m)
                    for m in range(interpolation_nodes)
                )
            )
        )

    def err_real(smearing, cutoff):
        return (
            prefac
            / torch.sqrt(cutoff * volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def loss(smearing, ns_mesh, cutoff):
        return torch.sqrt(
            err_Fourier(smearing, ns_mesh) ** 2 + err_real(smearing, cutoff) ** 2
        )

    params = [smearing_opt, ns_mesh_opt, cutoff_opt]
    _optimize_parameters(
        params=params,
        loss=loss,
        max_steps=max_steps,
        accuracy=accuracy,
        learning_rate=learning_rate,
    )

    return (
        float(smearing_opt),
        {
            "mesh_spacing": float(torch.min(cell_dimensions / ns_mesh_opt)),
            "interpolation_nodes": int(interpolation_nodes),
        },
        float(cutoff_opt),
    )
