from typing import Optional

from ..calculators.ewaldpotential import EwaldPotential as EwaldPotentialTorch
from .base import CalculatorBaseMetatensor


class EwaldPotential(CalculatorBaseMetatensor):
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
