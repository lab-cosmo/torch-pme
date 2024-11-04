from .. import calculators as torch_calculators
from .calculator import Calculator


class EwaldCalculator(Calculator):
    r"""
    Potential computed using the Ewald sum.

    Refer to :class:`torchpme.EwaldCalculator` for parameter documentation.

    For an **example** on the usage for any calculator refer to :ref:`userdoc-how-to`.
    """

    # see torchpme.metatensor.calculator
    _base_calculator = torch_calculators.EwaldCalculator
