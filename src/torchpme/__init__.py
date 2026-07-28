import contextlib

from . import calculators, lib, potentials, prefactors, tuning  # noqa
from ._version import __version__, __version_tuple__  # noqa
from .calculators import (
    Calculator,
    CalculatorDipole,
    EwaldCalculator,
    P3MCalculator,
    PMECalculator,
)
from .potentials import (
    CombinedPotential,
    CoulombPotential,
    InversePowerLawPotential,
    Potential,
    PotentialDipole,
    SplinePotential,
)

with contextlib.suppress(ImportError):
    from . import metatensor  # noqa

__all__ = [
    "Calculator",
    "CalculatorDipole",
    "CombinedPotential",
    "CoulombPotential",
    "EwaldCalculator",
    "InversePowerLawPotential",
    "P3MCalculator",
    "PMECalculator",
    "Potential",
    "PotentialDipole",
    "SplinePotential",
]
