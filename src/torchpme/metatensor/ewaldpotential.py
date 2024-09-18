from typing import Optional

from ..calculators.ewaldpotential import _EwaldPotentialImpl
from .base import CalculatorBaseMetatensor


class EwaldPotential(CalculatorBaseMetatensor, _EwaldPotentialImpl):
    r"""Potential computed using the Ewald sum.

    Refer to :class:`torchpme.EwaldPotential` for parameter documentation.

    For an **example** on the usage refer to :py:class:`metatensor.PMEPotential
    <torchpme.metatensor.PMEPotential>`.
    """

    def __init__(
        self,
        exponent: float = 1.0,
        atomic_smearing: Optional[float] = None,
        lr_wavelength: Optional[float] = None,
        subtract_self: bool = True,
        subtract_interior: bool = False,
    ):
        _EwaldPotentialImpl.__init__(
            self,
            exponent=exponent,
            atomic_smearing=atomic_smearing,
            lr_wavelength=lr_wavelength,
            subtract_self=subtract_self,
            subtract_interior=subtract_interior,
        )
        CalculatorBaseMetatensor.__init__(self)
