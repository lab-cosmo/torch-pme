import contextlib

from . import calculators, lib, potentials, utils  # noqa
from ._version import __version__, __version_tuple__  # noqa
from .calculators import Calculator, EwaldCalculator, P3MCalculator, PMECalculator
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
    "P3MCalculator",
    "PMECalculator",
    "CoulombPotential",
    "Potential",
    "InversePowerLawPotential",
    "SplinePotential",
    "CombinedPotential",
]
