import math
import warnings
from typing import Literal, Optional

import torch

from validation import _validate_forward_parameters


def estimate_smearing(
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


def tune_ewald(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    exponent: int = 1,
    accuracy: Optional[Literal["fast", "medium", "accurate"] | float] = "fast",
    max_steps: int = 50000,
    learning_rate: float = 5e-2,
    verbose: bool = False,
) -> tuple[dict[str, float], float]:
    r"""Find the optimal parameters for a single system for the ewald method.

    For the error formulas are given `elsewhere <https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/4/4d/Script_Longrange_Interactions.pdf>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha &= \left( \sqrt{2}\,\mathrm{smearing} \right)^{-1}

        K &= \frac{2 \pi}{\mathrm{lr\_wavelength}}

        r_c &= \mathrm{cutoff}

    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param charges: single tensor of shape (``1, len(positions))``.
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param exponent: exponent :math:`p` in :math:`1/r^p` potentials
    :param accuracy: Mode used to determine the optimal parameters. Possible values are
        ``"fast"``, ``"medium"`` or ``"accurate"``. For ``"fast"`` the parameters are
        set based on the number of atoms in the system to achieve a scaling of
        :math:`\mathcal{O}(N^{3/2})`. For ``"medium"`` or ``"accurate"``, the parameters
        are optimized using gradient descent until an estimated error of :math:`10^{-3}`
        or :math:`10^{-6}` is reached.
        Instead of ``"fast"``, ``"medium"`` or ``"accurate"``, you can give a float
        value for the accuracy.
    :param max_steps: maximum number of gradient descent steps
    :param learning_rate: learning rate for gradient descent
    :param verbose: whether to print the progress of gradient descent

    :return: Tuple containing a dictionary with the parameters for
        :py:class:`CalculatorEwald` and a float of the optimal cutoff value for the
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
    >>> ewald_parameter, cutoff = tune_ewald(charges, cell, positions, accuracy="fast")

    You can check the values of the parameters

    >>> print(ewald_parameter)
    {'smearing': 1.0318106837793297, 'lr_wavelength': 2.9468444218696392}

    >>> print(cutoff)
    2.2699835043145256

    Which can be used to initilize an :py:class:`CalculatorEwald` instance with
    parameters that are optimal for the system.
    """

    if exponent != 1:
        raise NotImplementedError("Only exponent = 1 is supported")

    dtype = positions.dtype
    device = positions.device

    # Create valid dummy tensors to verify `positions`, `charges` and `cell`
    neighbor_indices = torch.zeros(0, 2, device=device)
    neighbor_distances = torch.zeros(0, device=device)
    _validate_forward_parameters(
        charges=charges,
        cell=cell,
        positions=positions,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )

    if charges.shape[1] > 1:
        raise NotImplementedError(
            f"Found {charges.shape[1]} charge channels, but only one iss supported"
        )

    if accuracy == "fast":
        # The factors below are chosen to achieve an additional improved balance
        # between accuracy and speed, while maintaining a N^3/2 scaling. The values
        # result from tests on a CsCl system, whose unit cell is repeated 16 times
        # in each direction, resulting in a system of 8192 atoms.
        smearing_factor = 1.3
        lr_wavelength_factor = 2.2

        smearing = smearing_factor * len(positions) ** (1 / 6) / 2**0.5

        return {
            "smearing": smearing,
            "lr_wavelength": 2 * torch.pi * smearing / lr_wavelength_factor,
        }, smearing * lr_wavelength_factor

    if accuracy == "medium":
        accuracy = 1e-3
    elif accuracy == "accurate":
        accuracy = 1e-6
    elif not isinstance(accuracy, float):
        raise ValueError(
            f"'{accuracy}' is not a valid method or a float: Choose from 'fast',"
            f"'medium' or 'accurate', or provide a float for the accuracy."
        )

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = float(torch.min(cell_dimensions) / 2)

    smearing_init = estimate_smearing(cell)
    prefac = 2 * torch.sum(charges**2) / math.sqrt(len(positions))
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
            err_Fourier(smearing, lr_wavelength) ** 2 + err_real(smearing, cutoff) ** 2
        )

    # initial guess
    smearing = torch.tensor(
        smearing_init, device=device, dtype=dtype, requires_grad=True
    )
    lr_wavelength = torch.tensor(
        half_cell, device=device, dtype=dtype, requires_grad=True
    )
    cutoff = torch.tensor(
        half_cell / 10, device=device, dtype=dtype, requires_grad=True
    )

    optimizer = torch.optim.Adam([smearing, lr_wavelength, cutoff], lr=learning_rate)

    for step in range(max_steps):
        loss_value = loss(smearing, smooth_lr_wavelength(lr_wavelength), cutoff)
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

    if loss_value > accuracy:
        warnings.warn(
            "The searching for the parameters is ended, but the error is "
            f"{float(loss_value):.3e}, larger than the given accuracy {accuracy}. "
            "Consider increase max_step and",
            stacklevel=2,
        )

    return {
        "smearing": float(smearing),
        "lr_wavelength": float(smooth_lr_wavelength(lr_wavelength)),
    }, float(cutoff)
