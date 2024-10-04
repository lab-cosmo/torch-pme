from typing import Literal, Optional

from .. import calculators as torch_calculators
from .base import Calculator

try:
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for torchpme.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    ) from None


class EwaldCalculator(Calculator):
    r"""
    Potential computed using the Ewald sum.

    Refer to :class:`torchpme.EwaldPotential` for parameter documentation.

    For an **example** on the usage refer to :py:class:`metatensor.PMEPotential
    <torchpme.metatensor.PMEPotential>`.
    """

    # see torchpme.metatensor.base
    _base_calculator = torch_calculators.PMECalculator


def tune_ewald(
    system: System,
    exponent: int = 1,
    accuracy: Optional[Literal["fast", "medium", "accurate"] | float] = "fast",
    max_steps: int = 50000,
    learning_rate: float = 5e-2,
    verbose: bool = False,
) -> tuple[dict[str, float], float]:
    """Find the optimal parameters for a single system for the ewald method.

    Refer to :class:`torchpme.tune_ewald` for parameter documentation.
    """

    return torch_calculators.ewald.tune_ewald(
        positions=system.positions,
        charges=system.get_data("charges").values,
        cell=system.cell,
        exponent=exponent,
        accuracy=accuracy,
        max_steps=max_steps,
        learning_rate=learning_rate,
        verbose=verbose,
    )
