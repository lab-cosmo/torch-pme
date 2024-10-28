import contextlib

from . import (
    calculators,  # noqa
    lib,  # noqa
    potentials,  # noqa
    utils,  # noqa
)
from .calculators import Calculator, EwaldCalculator, PMECalculator
from .potentials import CoulombPotential, InversePowerLawPotential

with contextlib.suppress(ImportError):
    from . import metatensor  # noqa

__all__ = [
    "Calculator",
    "EwaldCalculator",
    "PMECalculator",
    "CoulombPotential",
    "InversePowerLawPotential",
]
__version__ = "0.0.0-dev"
