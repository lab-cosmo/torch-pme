import contextlib

from . import calculators, lib, potentials, utils  # noqa
from .calculators import Calculator, EwaldCalculator, PMECalculator
from .potentials import (
    CombinedPotential,
    CoulombPotential,
    InversePowerLawPotential,
    Potential,
    SplinePotential,
)

with contextlib.suppress(ImportError):
    from . import metatensor  # noqa

__all__ = [
    "Calculator",
    "EwaldCalculator",
    "PMECalculator",
    "CoulombPotential",
    "Potential",
    "InversePowerLawPotential",
    "SplinePotential",
    "CombinedPotential",
]
__version__ = "0.0.0-dev"
