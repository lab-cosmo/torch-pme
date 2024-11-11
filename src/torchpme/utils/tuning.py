import math
import warnings
from typing import Callable, Optional

import torch


def _optimize_parameters(
    params: list[torch.Tensor],
    loss: Callable,
    max_steps: int = 10000,
    accuracy: float = 1e-6,
    learning_rate: float = 5e-3,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return params[0], params[1], params[2]


def _smooth_mesh_spacing(mesh_spacing, min_dimension):
    """
    Confine to (0, min_dimension), ensuring that the ``ns``
    parameter is not smaller than 1
    (see :py:func:`_compute_lr` of :py:class:`PMEPotential`).
    """
    return min_dimension * torch.sigmoid(mesh_spacing)


def _inverse_smooth_mesh_spacing(value, min_dimension):
    """smooth_mesh_spacing(inverse_smooth_mesh_spacing(value)) == value"""
    return -math.log(min_dimension / value - 1)


def tune_ewald(
    sum_squared_charges: float,
    cell: torch.Tensor,
    positions: torch.Tensor,
    smearing: Optional[float] = None,
    lr_wavelength: Optional[float] = None,
    cutoff: Optional[float] = None,
    exponent: int = 1,
    accuracy: float = 1e-3,
    max_steps: int = 50000,
    learning_rate: float = 5e-2,
    verbose: bool = False,
) -> tuple[float, dict[str, float], float]:
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

    For the optimization we use the :class:`torch.optim.Adam` optimizer. By default this
    function optimize the ``smearing``, ``lr_wavelength`` and ``cutoff`` based on the
    error formula given `online`_. You can limit the optimization by giving one or more
    parameters to the function. For example in usual ML workflows the cutoff is fixed
    and one wants to optimize only the ``smearing`` and the ``lr_wavelength`` with
    respect to the minimal error and fixed cutoff.

    .. hint::

        Tuning uses an initial guess for the optimization, which can be applied by
        setting ``max_steps = 0``. This can be useful if fast tuning is required. These
        values typically result in accuracies around :math:`10^{-7}`.


    :param sum_squared_charges: accumulated squared charges, must be positive
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param smearing: if its value is given, it will not be tuned, see
        :class:`torchpme.EwaldCalculator` for details
    :param lr_wavelength: if its value is given, it will not be tuned, see
        :class:`torchpme.EwaldCalculator` for details
    :param cutoff: if its value is given, it will not be tuned, see
        :class:`torchpme.EwaldCalculator` for details
    :param exponent: exponent :math:`p` in :math:`1/r^p` potentials
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.
    :param max_steps: maximum number of gradient descent steps
    :param learning_rate: learning rate for gradient descent
    :param verbose: whether to print the progress of gradient descent

    :return: Tuple containing a float of the optimal smearing for the :class:
        `CoulombPotential`, a dictionary with the parameters for
        :class:`EwaldCalculator` and a float of the optimal cutoff value for the
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

    You can give one parameter to the function to tune only other parameters, for
    example, fixing the cutoff to 0.1

    >>> smearing, parameter, cutoff = tune_ewald(
    ...     torch.sum(charges**2, dim=0), cell, positions, cutoff=0.1, accuracy=1e-1
    ... )

    You can check the values of the parameters, now the cutoff is fixed

    >>> print(smearing)
    0.03234481782822382

    >>> print(parameter)
    {'lr_wavelength': 0.004985734847925747}

    >>> print(cutoff)
    0.1

    """
    _validate_parameters(sum_squared_charges, cell, positions, exponent)

    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = float(torch.min(cell_dimensions) / 2)

    def inverse_smooth_lr_wavelength(value=half_cell / 10):
        """smooth_lr_wavelength(inverse_smooth_lr_wavelength(value)) == value"""
        return -math.log(min_dimension / value - 1)

    def estimate_cutoff():
        return half_cell

    smearing_init = _estimate_smearing(cell) if smearing is None else smearing
    lr_wavelength_init = (
        inverse_smooth_lr_wavelength()
        if lr_wavelength is None
        else inverse_smooth_lr_wavelength(lr_wavelength)
    )
    cutoff_init = estimate_cutoff() if cutoff is None else cutoff
    prefac = 2 * sum_squared_charges / math.sqrt(len(positions))
    volume = torch.abs(cell.det())

    def smooth_lr_wavelength(lr_wavelength):
        """
        Confine to (0, min_dimension), ensuring that the ``ns``
        parameter is not smaller than 1
        (see :func:`_compute_lr` of :class:`CalculatorEwald`).
        """
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
    # If a parameter is not given, it is initialized with an initial guess and needs
    # to be optimized
    smearing = torch.tensor(
        smearing_init,
        device=device,
        dtype=dtype,
        requires_grad=(smearing is None),
    )
    lr_wavelength = torch.tensor(
        lr_wavelength_init,
        device=device,
        dtype=dtype,
        requires_grad=(lr_wavelength is None),
    )
    cutoff = torch.tensor(
        cutoff_init, device=device, dtype=dtype, requires_grad=(cutoff is None)
    )

    smearing_opt, lr_wavelength_opt, cutoff_opt = _optimize_parameters(
        [smearing, lr_wavelength, cutoff],
        loss,
        max_steps,
        accuracy,
        learning_rate,
        verbose,
    )

    return (
        float(smearing_opt),
        {"lr_wavelength": float(smooth_lr_wavelength(lr_wavelength_opt))},
        float(cutoff_opt),
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
    learning_rate: float = 5e-3,
    verbose: bool = False,
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

    .. hint::

        Tuning uses an initial guess for the optimization, which can be applied by
        setting ``max_steps = 0``. This can be useful if fast tuning is required. These
        values typically result in accuracies around :math:`10^{-2}`.

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
    :param verbose: whether to print the progress of gradient descent

    :return: Tuple containing a float of the optimal smearing for the :class:
        `CoulombPotential`, a dictionary with the parameters for
        :class:`PMECalculator` and a float of the optimal cutoff value for the
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

    You can give one parameter to the function to tune only other parameters, for
    example, fixing the cutoff to 0.1

    >>> smearing, parameter, cutoff = tune_pme(
    ...     torch.sum(charges**2, dim=0), cell, positions, cutoff=0.1, accuracy=1e-1
    ... )

    You can check the values of the parameters, now the cutoff is fixed

    >>> print(smearing)
    0.024764025655599434

    >>> print(parameter)
    {'mesh_spacing': 0.012499975000000003, 'interpolation_nodes': 4}

    >>> print(cutoff)
    0.1

    """
    _validate_parameters(sum_squared_charges, cell, positions, exponent)

    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")
    interpolation_nodes = torch.tensor(interpolation_nodes)

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = float(torch.min(cell_dimensions) / 2)

    smearing_init = _estimate_smearing(cell) if smearing is None else smearing
    mesh_spacing_init = (
        _inverse_smooth_mesh_spacing(smearing_init / 8, min_dimension)
        if mesh_spacing is None
        else _inverse_smooth_mesh_spacing(mesh_spacing, min_dimension)
    )
    cutoff_init = half_cell / 5 if cutoff is None else cutoff

    prefac = 2 * sum_squared_charges / math.sqrt(len(positions))
    volume = torch.abs(cell.det())

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
            err_Fourier(smearing, _smooth_mesh_spacing(mesh_spacing, min_dimension))
            ** 2
            + err_real(smearing, cutoff) ** 2
        )

    # initial guess
    dtype = positions.dtype
    device = positions.device

    # If a parameter is not given, it is initialized with an initial guess and needs
    # to be optimized
    smearing = torch.tensor(
        smearing_init, device=device, dtype=dtype, requires_grad=(smearing is None)
    )
    mesh_spacing = torch.tensor(
        mesh_spacing_init,
        device=device,
        dtype=dtype,
        requires_grad=(mesh_spacing is None),
    )
    cutoff = torch.tensor(
        cutoff_init, device=device, dtype=dtype, requires_grad=(cutoff is None)
    )

    smearing_opt, mesh_spacing_opt, cutoff_opt = _optimize_parameters(
        [smearing, mesh_spacing, cutoff],
        loss,
        max_steps,
        accuracy,
        learning_rate,
        verbose,
    )

    return (
        float(smearing_opt),
        {
            "mesh_spacing": float(
                _smooth_mesh_spacing(mesh_spacing_opt, min_dimension)
            ),
            "interpolation_nodes": int(interpolation_nodes),
        },
        float(cutoff_opt),
    )


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
    verbose: bool = False,
):
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
    >>> from vesin.torch import NeighborList
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
    0.04171745838080964

    >>> print(parameter)
    {'mesh_spacing': 0.011998118077739114, 'interpolation_nodes': 4}

    >>> print(cutoff)
    0.15409232806437623

    """
    _validate_parameters(sum_squared_charges, cell, positions, exponent)

    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")
    interpolation_nodes = torch.tensor(interpolation_nodes)

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = float(torch.min(cell_dimensions) / 2)

    smearing_init = _estimate_smearing(cell) if smearing is None else smearing
    mesh_spacing_init = (
        _inverse_smooth_mesh_spacing(smearing_init / 8, min_dimension)
        if mesh_spacing is None
        else _inverse_smooth_mesh_spacing(mesh_spacing, min_dimension)
    )
    cutoff_init = half_cell if cutoff is None else cutoff
    prefac = 2 * sum_squared_charges / math.sqrt(len(positions))
    volume = torch.abs(cell.det())

    def err_Fourier(smearing, mesh_spacing):
        ns_mesh = _get_ns_mesh_differentiable(cell, mesh_spacing)
        inverse_cell = torch.linalg.inv(cell)
        reciprocal_cell = 2 * torch.pi * inverse_cell.T
        reciprocal_cell_dimensions = torch.linalg.norm(reciprocal_cell, dim=1)
        spacing = reciprocal_cell_dimensions / ns_mesh
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

    def loss(smearing, mesh_spacing, cutoff):
        return torch.sqrt(
            err_Fourier(smearing, _smooth_mesh_spacing(mesh_spacing, min_dimension))
            ** 2
            + err_real(smearing, cutoff) ** 2
        )

    # initial guess
    dtype = positions.dtype
    device = positions.device

    # If a parameter is not given, it is initialized with an initial guess and needs
    # to be optimized
    smearing = torch.tensor(
        smearing_init, device=device, dtype=dtype, requires_grad=(smearing is None)
    )
    mesh_spacing = torch.tensor(
        mesh_spacing_init,
        device=device,
        dtype=dtype,
        requires_grad=(mesh_spacing is None),
    )
    cutoff = torch.tensor(
        cutoff_init, device=device, dtype=dtype, requires_grad=(cutoff is None)
    )

    smearing_opt, mesh_spacing_opt, cutoff_opt = _optimize_parameters(
        [smearing, mesh_spacing, cutoff],
        loss,
        max_steps,
        accuracy,
        learning_rate,
        verbose,
    )

    return (
        float(smearing_opt),
        {
            "mesh_spacing": float(
                _smooth_mesh_spacing(mesh_spacing_opt, min_dimension)
            ),
            "interpolation_nodes": int(interpolation_nodes),
        },
        float(cutoff_opt),
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


def _validate_parameters(
    sum_squared_charges: float,
    cell: torch.Tensor,
    positions: torch.Tensor,
    exponent: int,
):
    if sum_squared_charges <= 0:
        raise ValueError(
            f"sum of squared charges must be positive, got {sum_squared_charges}"
        )
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
    """differentiable version of :func:`get_ns_mesh`"""
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
