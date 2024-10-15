from .. import calculators as torch_calculators
from .base import Calculator


class EwaldCalculator(Calculator):
    r"""
    Potential computed using the Ewald sum.

    Refer to :class:`torchpme.EwaldCalculator` for parameter documentation.

    For an **example** on the usage for any calculator refer to :ref:`userdoc-how-to`.
    """

    # see torchpme.metatensor.base
    _base_calculator = torch_calculators.EwaldCalculator
