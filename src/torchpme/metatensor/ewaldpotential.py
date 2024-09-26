from typing import Literal, Optional, Union

import torch

from ..calculators.ewaldpotential import EwaldPotential as EwaldPotentialTorch
from ..calculators.ewaldpotential import tune_ewald as tune_ewald_torch
from .base import CalculatorBaseMetatensor

try:
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for torchpme.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    ) from None


class EwaldPotential(CalculatorBaseMetatensor):
    r"""
    Potential computed using the Ewald sum.

    Refer to :class:`torchpme.EwaldPotential` for parameter documentation.

    For an **example** on the usage refer to :py:class:`metatensor.PMEPotential
    <torchpme.metatensor.PMEPotential>`.
    """

    def __init__(
        self,
        atomic_smearing: Union[float, torch.Tensor],
        lr_wavelength: float,
        exponent: float = 1.0,
        subtract_interior: bool = False,
        full_neighbor_list: bool = False,
    ):
        super().__init__()
        self.calculator = EwaldPotentialTorch(
            exponent=exponent,
            atomic_smearing=atomic_smearing,
            lr_wavelength=lr_wavelength,
            subtract_interior=subtract_interior,
            full_neighbor_list=full_neighbor_list,
        )


def tune_ewald(
    system: System,
    exponent: int = 1,
    method: Optional[Literal["fast", "medium", "accurate"]] = "fast",
    accuracy: Optional[float] = None,
    max_steps: int = 50000,
    learning_rate: float = 5e-2,
    verbose: bool = False,
) -> tuple[dict[str, float], float]:
    """Find the optimal parameters for a single system for the ewald method.

    Refer to :class:`torchpme.tune_ewald` for parameter documentation.
    """

    return tune_ewald_torch(
        positions=system.positions,
        charges=system.get_data("charges").values,
        cell=system.cell,
        exponent=exponent,
        method=method,
        accuracy=accuracy,
        max_steps=max_steps,
        learning_rate=learning_rate,
        verbose=verbose,
    )
