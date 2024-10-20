import contextlib

from . import utils  # noqa
from .calculators.base import Calculator
from .calculators.ewald import EwaldCalculator
from .calculators.pme import PMECalculator
from .lib.potentials import CoulombPotential, InversePowerLawPotential, CombinedPotential

with contextlib.suppress(ImportError):
    from . import metatensor  # noqa

__all__ = [
    "Calculator",
    "EwaldCalculator",
    "PMECalculator",
    "CoulombPotential",
    "InversePowerLawPotential",
    "CombinedPotential",
]
__version__ = "0.0.0-dev"
