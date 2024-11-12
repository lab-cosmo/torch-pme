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
) -> None:
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    for step in range(max_steps):
        loss_value = loss(*params)
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


class _SmoothLRWaveLength:
    """TODO: Explain why this is necessary."""

    def __init__(self, min_dimension: float):
        self.min_dimension = min_dimension

    def __call__(self, lr_wavelength):
        return -torch.log(self.min_dimension / lr_wavelength - 1)

    def inverse(self, smooth_lr_wavelength):
        return self.min_dimension * torch.sigmoid(smooth_lr_wavelength)


def _initial_guess(
    cell: torch.Tensor,
    smearing: Optional[float] = None,
    lr_wavelength: Optional[float] = None,
    cutoff: Optional[float] = None,
    accuracy: float = 1e-3,
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Initial guess for tuning.

    **smearing**: TODO

    **real_space**: Since the real-space sum is truncated at the cutoff, the contribution of
    an atom right outside of it would precisely be V_SR(cutoff), where V_SR is the
    short-range function. Hence, we pick as out initial guess simply the cutoff for
    which V_SR(cutoff) = accuracy.

    **reciprocal_space**: Since the reciprocal space sum is truncated at a cutoff kcut, the
    contribution of the next term in the sum would precisely be G_LR(kcut), where G_LR
    is the Fourier-transformed long-range function, also called the kernel. Hence, we
    pick as out initial guess simply the cutoff for which G_LR(kcut) = accuracy. In
    practice, we work with k squared rather than k.

    :param cell: A 3x3 tensor representing the periodic system
    :param accuracy: Recomended values for a balance between the accuracy and speed is
        :math:`10^{-3}`. For more accurate results, use :math:`10^{-6}`.
    :returns: estimation for ``smearing``, and ``cutoff``, and
        ``lr_wavelength``/``mesh_spacing``
    """
    if not isinstance(accuracy, float):
        raise ValueError(f"'{accuracy}' is not a float.")

    dtype = cell.dtype
    device = cell.device

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = torch.min(cell_dimensions)
    half_cell = float(min_dimension / 2)

    # estimate smearing
    if smearing is None:
        smearing_init = (half_cell - 1e-6) / 5.0

    smearing_init = torch.tensor(
        smearing_init if smearing is None else smearing,
        dtype=dtype,
        device=device,
        requires_grad=(smearing is None),
    )

    # estimate cutoff
    if cutoff is None:

        def loss_cutoff(cutoff):
            return torch.erfc(cutoff / math.sqrt(2) / smearing_init) / cutoff - accuracy

        cutoff_init = torch.tensor(
            half_cell, dtype=dtype, device=device, requires_grad=True
        )
        _optimize_parameters(params=[cutoff_init], loss=loss_cutoff)

    cutoff_init = torch.tensor(
        float(cutoff_init) if cutoff is None else cutoff,
        dtype=dtype,
        device=device,
        requires_grad=(cutoff is None),
    )

    # estimate lr_wavelength
    if lr_wavelength is None:

        def loss_k_sq(k_sq):
            return (
                torch.exp(-0.5 * smearing_init**2 * k_sq) * 4 * torch.pi / k_sq
                - accuracy
            )

        kcut_sq_init = torch.tensor(
            (2 * torch.pi / half_cell) ** 2,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        _optimize_parameters(params=[kcut_sq_init], loss=loss_k_sq)
        lr_wavelength_init = torch.sqrt(2 * torch.pi / kcut_sq_init)

    lr_wavelength_init = torch.tensor(
        float(lr_wavelength_init) if lr_wavelength is None else lr_wavelength,
        dtype=dtype,
        device=device,
        requires_grad=(lr_wavelength is None),
    )

    return smearing_init, lr_wavelength_init, cutoff_init


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

    if exponent != 1:
        raise NotImplementedError("Only exponent = 1 is supported")

    if list(positions.shape) != [len(positions), 3]:
        raise ValueError(
            "each `positions` must be a tensor with shape [n_atoms, 3], got at least "
            f"one tensor with shape {list(positions.shape)}"
        )

    # check shape, dtype and device of cell
    if list(cell.shape) != [3, 3]:
        raise ValueError(
            "each `cell` must be a tensor with shape [3, 3], got at least one tensor "
            f"with shape {list(cell.shape)}"
        )

    if torch.equal(cell.det(), torch.full([], 0, dtype=cell.dtype, device=cell.device)):
        raise ValueError(
            "provided `cell` has a determinant of 0 and therefore is not valid for "
            "periodic calculation"
        )

    dtype = positions.dtype
    if cell.dtype != dtype:
        raise ValueError(
            f"each `cell` must have the same type {dtype} as `positions`, got at least "
            "one tensor of type "
            f"{cell.dtype}"
        )

    device = positions.device
    if cell.device != device:
        raise ValueError(
            f"each `cell` must be on the same device {device} as `positions`, got at "
            "least one tensor with device "
            f"{cell.device}"
        )
