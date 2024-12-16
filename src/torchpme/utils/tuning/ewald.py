import math
from typing import Optional

import torch

from . import (
    _estimate_smearing_cutoff,
    _optimize_parameters,
    _validate_parameters,
)

TWO_PI = 2 * math.pi


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
    learning_rate: float = 0.1,
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

    :return: Tuple containing a float of the optimal smearing for the :class:
        `CoulombPotential`, a dictionary with the parameters for
        :class:`EwaldCalculator` and a float of the optimal cutoff value for the
        neighborlist computation.

    Example
    -------
    >>> import torch
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
    0.7527865828476816

    >>> print(parameter)
    {'lr_wavelength': 11.138556788117427}

    >>> print(cutoff)
    2.207855328192979

    You can give one parameter to the function to tune only other parameters, for
    example, fixing the cutoff to 0.1

    >>> smearing, parameter, cutoff = tune_ewald(
    ...     torch.sum(charges**2, dim=0), cell, positions, cutoff=0.4, accuracy=1e-1
    ... )

    You can check the values of the parameters, now the cutoff is fixed

    >>> print(round(smearing, 4))
    0.1402

    We can also check the value of the other parameter like the ``lr_wavelength``

    >>> print(round(parameter["lr_wavelength"], 3))
    0.255

    and finally as requested the value of the cutoff is fixed

    >>> print(cutoff)
    0.4

    """
    _validate_parameters(sum_squared_charges, cell, positions, exponent, accuracy)

    smearing_opt, cutoff_opt = _estimate_smearing_cutoff(
        cell=cell, smearing=smearing, cutoff=cutoff, accuracy=accuracy
    )

    # We choose a very small initial fourier wavelength, hardcoded for now
    k_cutoff_opt = torch.tensor(
        1e-3 if lr_wavelength is None else TWO_PI / lr_wavelength,
        dtype=cell.dtype,
        device=cell.device,
        requires_grad=(lr_wavelength is None),
    )

    volume = torch.abs(cell.det())
    prefac = 2 * sum_squared_charges / math.sqrt(len(positions))

    def err_Fourier(smearing, k_cutoff):
        return (
            prefac**0.5
            / smearing
            / torch.sqrt(TWO_PI**2 * volume / (TWO_PI / k_cutoff) ** 0.5)
            * torch.exp(-(TWO_PI**2) * smearing**2 / (TWO_PI / k_cutoff))
        )

    def err_real(smearing, cutoff):
        return (
            prefac
            / torch.sqrt(cutoff * volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def loss(smearing, k_cutoff, cutoff):
        return torch.sqrt(
            err_Fourier(smearing, k_cutoff) ** 2 + err_real(smearing, cutoff) ** 2
        )

    params = [smearing_opt, k_cutoff_opt, cutoff_opt]
    _optimize_parameters(
        params=params,
        loss=loss,
        max_steps=max_steps,
        accuracy=accuracy,
        learning_rate=learning_rate,
    )

    return (
        float(smearing_opt),
        {"lr_wavelength": TWO_PI / float(k_cutoff_opt)},
        float(cutoff_opt),
    )
