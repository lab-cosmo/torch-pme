from typing import Literal, Optional

import torch

from .. import calculators as torch_calculators
from .base import Calculator

try:
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for torchpme.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    ) from None


class EwaldCalculator(Calculator):
    r"""
    Potential computed using the Ewald sum.

    Refer to :class:`torchpme.EwaldCalculator` for parameter documentation.

    For an **example** on the usage for any calculator refer to :ref:`userdoc-how-to`.
    """

    # see torchpme.metatensor.base
    _base_calculator = torch_calculators.EwaldCalculator
