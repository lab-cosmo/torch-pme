import math
from typing import Optional

import torch

from ...lib import get_ns_mesh
from . import (
    _estimate_smearing_cutoff,
    _optimize_parameters,
    _validate_parameters,
)


def tune_pme(
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
    learning_rate: float = 0.1,
):
    r"""
    Find the optimal parameters for :class:`torchpme.PMECalculator`.

    For the error formulas are given `elsewhere <https://doi.org/10.1063/1.470043>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    For the optimization we use the :class:`torch.optim.Adam` optimizer. By default this
    function optimize the ``smearing``, ``mesh_spacing`` and ``cutoff`` based on the
    error formula given `elsewhere`_. You can limit the optimization by giving one or
    more parameters to the function. For example in usual ML workflows the cutoff is
    fixed and one wants to optimize only the ``smearing`` and the ``mesh_spacing`` with
    respect to the minimal error and fixed cutoff.

    :param sum_squared_charges: accumulated squared charges, must be positive
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param smearing: if its value is given, it will not be tuned, see
        :class:`torchpme.PMECalculator` for details
    :param mesh_spacing: if its value is given, it will not be tuned, see
        :class:`torchpme.PMECalculator` for details
    :param cutoff: if its value is given, it will not be tuned, see
        :class:`torchpme.PMECalculator` for details
    :param interpolation_nodes: The number ``n`` of nodes used in the interpolation per
        coordinate axis. The total number of interpolation nodes in 3D will be ``n^3``.
        In general, for ``n`` nodes, the interpolation will be performed by piecewise
        polynomials of degree ``n - 1`` (e.g. ``n = 4`` for cubic interpolation). Only
        the values ``3, 4, 5, 6, 7`` are supported.
    :param exponent: exponent :math:`p` in :math:`1/r^p` potentials
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.
    :param max_steps: maximum number of gradient descent steps
    :param learning_rate: learning rate for gradient descent

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
    >>> smearing, parameter, cutoff = tune_pme(
    ...     torch.sum(charges**2, dim=0), cell, positions, accuracy=1e-1
    ... )

    You can check the values of the parameters

    >>> print(smearing)
    0.6700526796270038

    >>> print(parameter)
    {'mesh_spacing': 0.6332684025633143, 'interpolation_nodes': 4}

    >>> print(cutoff)
    2.175844455830708

    You can give one parameter to the function to tune only other parameters, for
    example, fixing the cutoff to 0.1

    >>> smearing, parameter, cutoff = tune_pme(
    ...     torch.sum(charges**2, dim=0), cell, positions, cutoff=0.6, accuracy=1e-1
    ... )

    You can check the values of the parameters, now the cutoff is fixed

    >>> print(smearing)
    0.20909349851660716

    >>> print(parameter)
    {'mesh_spacing': 0.16520200966949541, 'interpolation_nodes': 4}

    >>> print(cutoff)
    0.6

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

    interpolation_nodes = torch.tensor(interpolation_nodes, device=cell.device)

    err_bound = PMEErrorBounds(
        sum_squared_charges=sum_squared_charges, cell=cell, positions=positions
    )

    params = [cutoff_opt, smearing_opt, ns_mesh_opt, interpolation_nodes]
    _optimize_parameters(
        params=params,
        loss=err_bound,
        max_steps=max_steps,
        accuracy=accuracy,
        learning_rate=learning_rate,
    )

    return (
        float(smearing_opt),
        {
            "mesh_spacing": float(torch.min(cell_dimensions / ((ns_mesh_opt - 1) / 2))),
            "interpolation_nodes": int(interpolation_nodes),
        },
        float(cutoff_opt),
    )


class PMEErrorBounds(torch.nn.Module):
    def __init__(
        self, sum_squared_charges: float, cell: torch.Tensor, positions: torch.Tensor
    ):
        self.volume = torch.abs(torch.det(cell))
        self.prefac = 2 * sum_squared_charges / math.sqrt(len(positions))
        self.cell = cell
        self.positions = positions
        super().__init__()

    def err_kspace(self, smearing, ns_mesh, interpolation_nodes):
        cell_dimensions = torch.linalg.norm(self.cell, dim=1)
        H = torch.prod(cell_dimensions / ns_mesh) ** (1 / 3)
        print(H)
        i_n_factorial = torch.exp(torch.lgamma(interpolation_nodes + 1))
        RMS_phi = [None, None, 0.246, 0.404, 0.950, 2.51, 8.42]

        return (
            self.prefac
            * torch.pi**0.25
            * (6 * (1 / 2**0.5 / smearing) / (2 * interpolation_nodes + 1)) ** 0.5
            / self.volume ** (2 / 3)
            * (2**0.5 / smearing * H) ** interpolation_nodes
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

    def forward(self, cutoff, smearing, ns_mesh, interpolation_nodes):
        return torch.sqrt(
            self.err_rspace(smearing, cutoff) ** 2
            + self.err_kspace(smearing, ns_mesh, interpolation_nodes) ** 2
        )
