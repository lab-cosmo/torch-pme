from .. import calculators as torch_calculators
from .calculator import Calculator


class P3MCalculator(Calculator):
    r"""
    Potential using a particle-particle particle-mesh based Ewald (P3M).

    Refer to :class:`torchpme.P3MCalculator` for parameter documentation.

    For an **example** on the usage for any calculator refer to :ref:`userdoc-how-to`.
    """

    # see torchpme.metatensor.calculator
    _base_calculator = torch_calculators.P3MCalculator
