import math
from typing import Optional

import torch

from . import (
    _initial_guess,
    _optimize_parameters,
    _SmoothLRWaveLength,
    _validate_parameters,
)


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

    smearing_opt, lr_wavelength_opt, cutoff_opt = _initial_guess(
        cell=cell,
        smearing=smearing,
        lr_wavelength=lr_wavelength,
        cutoff=cutoff,
        accuracy=accuracy,
    )

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = torch.min(cell_dimensions)
    volume = torch.abs(cell.det())
    prefac = 2 * sum_squared_charges / math.sqrt(len(positions))

    smoother = _SmoothLRWaveLength(min_dimension)
    smooth_lr_wavelength_opt = smoother(lr_wavelength_opt)

    # we have to clear the graph before passing it to the optimizer
    smooth_lr_wavelength_opt = torch.tensor(
        float(smooth_lr_wavelength_opt),
        dtype=lr_wavelength_opt.dtype,
        device=lr_wavelength_opt.device,
        requires_grad=lr_wavelength_opt.requires_grad,
    )

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

    def loss(smearing, smooth_lr_wavelength, cutoff):
        return torch.sqrt(
            err_Fourier(smearing, smoother.inverse(smooth_lr_wavelength)) ** 2
            + err_real(smearing, cutoff) ** 2
        )

    params = [smearing_opt, smooth_lr_wavelength_opt, cutoff_opt]
    _optimize_parameters(
        params=params,
        loss=loss,
        max_steps=max_steps,
        accuracy=accuracy,
        learning_rate=learning_rate,
        verbose=verbose,
    )

    return (
        float(smearing_opt),
        {"lr_wavelength": float(smoother.inverse(smooth_lr_wavelength_opt))},
        float(cutoff_opt),
    )
