from typing import Literal, Optional

import torch

from .. import calculators as torch_calculators
from .base import Calculator

try:
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for torchpme.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    ) from None


class PMECalculator(Calculator):
    r"""
    Potential using a particle mesh-based Ewald (PME).

    Refer to :py:class:`torchpme.PMECalculator` for parameter documentation.

    For an **example** on the usage for any calculator refer to :ref:`userdoc-how-to`.
    """

    # see torchpme.metatensor.base
    _base_calculator = torch_calculators.PMECalculator


def tune_pme(
    system: System,
    interpolation_nodes: int = 4,
    exponent: int = 1,
    accuracy: Optional[Literal["medium", "accurate"] | float] = "medium",
    max_steps: int = 50000,
    learning_rate: float = 5e-3,
    verbose: bool = False,
) -> tuple[float, dict[str, float], float]:
    """Find the optimal parameters for a single system for the PME method.

    Refer to :class:`torchpme.tune_pme` for parameter documentation.
    """

    return torch_calculators.pme.tune_pme(
        positions=system.positions,
        sum_squared_charges=torch.sum(system.get_data("charges").values ** 2, dim=0),
        cell=system.cell,
        interpolation_nodes=interpolation_nodes,
        exponent=exponent,
        accuracy=accuracy,
        max_steps=max_steps,
        learning_rate=learning_rate,
        verbose=verbose,
    )
