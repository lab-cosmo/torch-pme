import math
import warnings
from typing import Callable

import torch


def _optimize_parameters(
    params: list[torch.Tensor],
    loss: Callable,
    max_steps: int = 10000,
    accuracy: float = 1e-6,
    learning_rate: float = 5e-3,
    verbose: bool = False,
):
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    for step in range(max_steps):
        loss_value = loss(params[0], params[1], params[2])
        if torch.isnan(loss_value):
            raise ValueError(
                "The value of the estimated error is now nan, consider using a "
                "smaller learning rate."
            )
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose and (step % 100 == 0):
            print(f"{step}: {loss_value:.2e}")
        if loss_value <= accuracy:
            break

    if max_steps == 0:
        warnings.warn(
            "Skip optimization, return the initial guess.",
            stacklevel=2,
        )
    elif loss_value > accuracy:
        warnings.warn(
            "The searching for the parameters is ended, but the error is "
            f"{float(loss_value):.3e}, larger than the given accuracy {accuracy}. "
            "Consider increase max_step and",
            stacklevel=2,
        )


def tune_ewald(
    sum_squared_charges: float,
    cell: torch.Tensor,
    positions: torch.Tensor,
    exponent: int = 1,
    accuracy: float = 1e-3,
    max_steps: int = 50000,
    learning_rate: float = 5e-2,
    verbose: bool = False,
) -> tuple[float, dict[str, float], float]:
    r"""
    Find the optimal parameters for :class:`torchpme.calculators.ewald.EwaldCalculator`.

    The error formulas are given `online <https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/4/4d/Script_Longrange_Interactions.pdf>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha &= \left( \sqrt{2}\,\mathrm{smearing} \right)^{-1}

        K &= \frac{2 \pi}{\mathrm{lr\_wavelength}}

        r_c &= \mathrm{cutoff}

    :param sum_squared_charges: accumulated squared charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param exponent: exponent :math:`p` in :math:`1/r^p` potentials
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.
    :param max_steps: maximum number of gradient descent steps
    :param learning_rate: learning rate for gradient descent
    :param verbose: whether to print the progress of gradient descent

    :return: Tuple containing a float of the optimal smearing for the :py:class:
        `CoulombPotential`, a dictionary with the parameters for :py:class:`EwaldCalculator` and a float of the optimal cutoff value for the
        neighborlist computation.

    Example
    -------
    >>> import torch
    >>> from vesin.torch import NeighborList
    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)
    >>> smearing, parameter, cutoff = tune_ewald(
    ...     torch.sum(charges**2, dim=0), cell, positions, accuracy=1e-1
    ... )

    You can check the values of the parameters

    >>> print(smearing)
    0.14999979983727296

    >>> print(parameter)
    {'lr_wavelength': 0.047677734917968666}

    >>> print(cutoff)
    0.5485209762493759
    """

    _validate_parameters(cell, positions, exponent)

    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = float(torch.min(cell_dimensions) / 2)

    smearing_init = _estimate_smearing(cell)
    prefac = 2 * sum_squared_charges / math.sqrt(len(positions))
    volume = torch.abs(cell.det())

    def smooth_lr_wavelength(lr_wavelength):
        """Confine to (0, min_dimension), ensuring that the ``ns``
        parameter is not smaller than 1
        (see :py:func:`_compute_lr` of :py:class:`CalculatorEwald`)."""
        return min_dimension * torch.sigmoid(lr_wavelength)

    def err_Fourier(smearing, lr_wavelength):
        return (
            prefac**0.5
            / smearing
            / torch.sqrt(2 * torch.pi**2 * volume / lr_wavelength**0.5)
            * torch.exp(-2 * torch.pi**2 * smearing**2 / lr_wavelength)
        )

    def err_real(smearing, cutoff):
        return (
            prefac
            / torch.sqrt(cutoff * volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def loss(smearing, lr_wavelength, cutoff):
        return torch.sqrt(
            err_Fourier(smearing, smooth_lr_wavelength(lr_wavelength)) ** 2
            + err_real(smearing, cutoff) ** 2
        )

    # initial guess
    dtype = positions.dtype
    device = positions.device

    smearing = torch.tensor(
        smearing_init, device=device, dtype=dtype, requires_grad=True
    )
    lr_wavelength = torch.tensor(
        -math.log(10 * min_dimension / half_cell - 1),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )  # sigmoid(lr_wavelength) == half_cell / 10
    cutoff = torch.tensor(half_cell, device=device, dtype=dtype, requires_grad=True)

    _optimize_parameters(
        [smearing, lr_wavelength, cutoff],
        loss,
        max_steps,
        accuracy,
        learning_rate,
        verbose,
    )

    return (
        float(smearing),
        {"lr_wavelength": float(smooth_lr_wavelength(lr_wavelength))},
        float(cutoff),
    )


def tune_pme(
    sum_squared_charges: float,
    cell: torch.Tensor,
    positions: torch.Tensor,
    interpolation_nodes: int = 4,
    exponent: int = 1,
    accuracy: float = 1e-3,
    max_steps: int = 50000,
    learning_rate: float = 5e-3,
    verbose: bool = False,
):
    r"""Find the optimal parameters for :class:`torchpme.calculators.pme.PMECalculator`.

    For the error formulas are given `elsewhere <https://doi.org/10.1063/1.470043>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    :param sum_squared_charges: accumulated squared charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
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
    :param verbose: whether to print the progress of gradient descent

    :return: Tuple containing a float of the optimal smearing for the :py:class:
        `CoulombPotential`, a dictionary with the parameters for
        :py:class:`PMECalculator` and a float of the optimal cutoff value for the
        neighborlist computation.

    Example
    -------
    >>> import torch
    >>> from vesin.torch import NeighborList
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
    0.04576166523476457

    >>> print(parameter)
    {'mesh_spacing': 0.012499975000000003, 'interpolation_nodes': 4}

    >>> print(cutoff)
    0.15078003506282253
    """

    _validate_parameters(cell, positions, exponent)

    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")
    interpolation_nodes = torch.tensor(interpolation_nodes)

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = float(torch.min(cell_dimensions) / 2)

    smearing_init = _estimate_smearing(cell)
    prefac = 2 * sum_squared_charges / math.sqrt(len(positions))
    volume = torch.abs(cell.det())

    def smooth_mesh_spacing(mesh_spacing):
        """Confine to (0, min_dimension), ensuring that the ``ns``
        parameter is not smaller than 1
        (see :py:func:`_compute_lr` of :py:class:`PMEPotential`)."""
        return min_dimension * torch.sigmoid(mesh_spacing)

    def err_Fourier(smearing, mesh_spacing):
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

        ns_mesh = _get_ns_mesh_differentiable(cell, mesh_spacing)

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

    def loss(smearing, mesh_spacing, cutoff):
        return torch.sqrt(
            err_Fourier(smearing, smooth_mesh_spacing(mesh_spacing)) ** 2
            + err_real(smearing, cutoff) ** 2
        )

    # initial guess
    dtype = positions.dtype
    device = positions.device

    smearing = torch.tensor(
        smearing_init, device=device, dtype=dtype, requires_grad=True
    )

    # smooth_mesh_spacing(mesh_spacing) = smearing / 8, is the standard initial guess
    mesh_spacing = torch.tensor(
        -math.log(min_dimension * 8 / smearing_init - 1),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    cutoff = torch.tensor(half_cell / 5, device=device, dtype=dtype, requires_grad=True)

    _optimize_parameters(
        [smearing, mesh_spacing, cutoff],
        loss,
        max_steps,
        accuracy,
        learning_rate,
        verbose,
    )

    return (
        float(smearing),
        {
            "mesh_spacing": float(smooth_mesh_spacing(mesh_spacing)),
            "interpolation_nodes": int(interpolation_nodes),
        },
        float(cutoff),
    )


def _estimate_smearing(
    cell: torch.Tensor,
) -> float:
    """
    Estimate the smearing for ewald calculators.

    :param cell: A 3x3 tensor representing the periodic system
    :returns: estimated smearing
    """
    if torch.equal(cell.det(), torch.full([], 0, dtype=cell.dtype, device=cell.device)):
        raise ValueError(
            "provided `cell` has a determinant of 0 and therefore is not valid "
            "for periodic calculation"
        )

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    max_cutoff = torch.min(cell_dimensions) / 2 - 1e-6

    return max_cutoff.item() / 5.0


def _validate_parameters(cell: torch.Tensor, positions: torch.Tensor, exponent: int):
    dtype = positions.dtype
    device = positions.device
    if exponent != 1:
        raise NotImplementedError("Only exponent = 1 is supported")

    if list(positions.shape) != [len(positions), 3]:
        raise ValueError(
            "each `positions` must be a tensor with shape [n_atoms, 3], got at "
            f"least one tensor with shape {list(positions.shape)}"
        )

    # check shape, dtype and device of cell
    if list(cell.shape) != [3, 3]:
        raise ValueError(
            "each `cell` must be a tensor with shape [3, 3], got at least "
            f"one tensor with shape {list(cell.shape)}"
        )

    if cell.dtype != dtype:
        raise ValueError(
            f"each `cell` must have the same type {dtype} as "
            "`positions`, got at least one tensor of type "
            f"{cell.dtype}"
        )

    if cell.device != device:
        raise ValueError(
            f"each `cell` must be on the same device {device} as "
            "`positions`, got at least one tensor with device "
            f"{cell.device}"
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
    positions_rel += (
        torch.randn(positions_rel.shape) * 1e-10
    )  # Noises help the algorithm work for tiny systems (<100 atoms)
    return (
        torch.mean(
            (torch.prod(indices_to_interpolate - positions_rel, dim=0)) ** 2, dim=0
        )
        ** 0.5
    )


def _get_ns_mesh_differentiable(cell: torch.Tensor, mesh_spacing: float):
    """differentiable version of :py:func:`get_ns_mesh`"""

    basis_norms = torch.linalg.norm(cell, dim=1)
    ns_approx = basis_norms / mesh_spacing
    ns_actual_approx = 2 * ns_approx + 1  # actual number of mesh points
    # ns = [nx, ny, nz], closest power of 2 (helps for FT efficiency)
    return torch.tensor(2).pow(_Ceil.apply(torch.log2(ns_actual_approx)))


class _Ceil(torch.autograd.Function):
    """ceil function with non-zero gradient"""

    @staticmethod
    def forward(ctx, input):
        result = torch.ceil(input)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


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
