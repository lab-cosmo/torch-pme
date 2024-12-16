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
    0.6768985898318037

    >>> print(parameter)
    {'mesh_spacing': 0.6305733973385922, 'interpolation_nodes': 4}

    >>> print(cutoff)
    2.243154348782357

    You can give one parameter to the function to tune only other parameters, for
    example, fixing the cutoff to 0.1

    >>> smearing, parameter, cutoff = tune_pme(
    ...     torch.sum(charges**2, dim=0), cell, positions, cutoff=0.6, accuracy=1e-1
    ... )

    You can check the values of the parameters, now the cutoff is fixed

    >>> print(smearing)
    0.22038829671671745

    >>> print(parameter)
    {'mesh_spacing': 0.5006356677116188, 'interpolation_nodes': 4}

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
    volume = torch.abs(cell.det())
    prefac = 2 * sum_squared_charges / math.sqrt(len(positions))

    interpolation_nodes = torch.tensor(interpolation_nodes, device=cell.device)

    def err_Fourier(smearing, ns_mesh):
        def H(ns_mesh):
            return torch.prod(1 / ns_mesh) ** (1 / 3)

        def RMS_phi(ns_mesh):
            return torch.linalg.norm(
                _compute_RMS_phi(cell, interpolation_nodes, ns_mesh, positions)
            )

        def log_factorial(x):
            return torch.lgamma(x + 1)

        def factorial(x):
            return torch.exp(log_factorial(x))

        return (
            prefac
            * torch.pi**0.25
            * (6 * (1 / 2**0.5 / smearing) / (2 * interpolation_nodes + 1)) ** 0.5
            / volume ** (2 / 3)
            * (2**0.5 / smearing * H(ns_mesh)) ** interpolation_nodes
            / factorial(interpolation_nodes)
            * torch.exp(
                (interpolation_nodes) * (torch.log(interpolation_nodes / 2) - 1) / 2
            )
            * RMS_phi(ns_mesh)
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


def _compute_RMS_phi(
    cell: torch.Tensor,
    interpolation_nodes: torch.Tensor,
    ns_mesh: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    inverse_cell = torch.linalg.inv(cell)
    # Compute positions relative to the mesh basis vectors
    positions_rel = ns_mesh * torch.matmul(positions, inverse_cell)

    # Calculate positions and distances based on interpolation nodes
    even = interpolation_nodes % 2 == 0
    if even:
        # For Lagrange interpolation, when the number of interpolation
        # is even, the relative position of a charge is the midpoint of
        # the two nearest gridpoints.
        positions_rel_idx = _Floor.apply(positions_rel)
    else:
        # For Lagrange interpolation, when the number of interpolation
        # points is odd, the relative position of a charge is the nearest gridpoint.
        positions_rel_idx = _Round.apply(positions_rel)

    # Calculate indices of mesh points on which the particle weights are
    # interpolated. For each particle, its weight is "smeared" onto `order**3` mesh
    # points, which can be achived using meshgrid below.
    indices_to_interpolate = torch.stack(
        [
            (positions_rel_idx + i)
            for i in range(
                1 - (interpolation_nodes + 1) // 2,
                1 + interpolation_nodes // 2,
            )
        ],
        dim=0,
    )
    positions_rel = positions_rel[torch.newaxis, :, :]
    positions_rel += 1e-10 * torch.randn(
        positions_rel.shape, dtype=cell.dtype, device=cell.device
    )  # Noises help the algorithm work for tiny systems (<100 atoms)
    return (
        torch.mean(
            (torch.prod(indices_to_interpolate - positions_rel, dim=0)) ** 2, dim=0
        )
        ** 0.5
    )


class _Floor(torch.autograd.Function):
    """floor function with non-zero gradient"""

    @staticmethod
    def forward(ctx, input):
        result = torch.floor(input)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _Round(torch.autograd.Function):
    """round function with non-zero gradient"""

    @staticmethod
    def forward(ctx, input):
        result = torch.round(input)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
