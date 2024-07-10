from typing import Optional

import torch

from ..calculators.ewaldpotential import _EwaldPotentialImpl
from .base import CalculatorBaseMetatensor


class EwaldPotential(CalculatorBaseMetatensor, _EwaldPotentialImpl):
    r"""Specie-wise long-range potential computed using the Ewald sum.

    Refer to :class:`meshlode.EwaldPotential` for parameter documentation.

    Example
    -------
    We calculate the Madelung constant of a CsCl (Cesium-Chloride) crystal. The
    reference value is :math:`2 \cdot 1.7626 / \sqrt{3} \approx 2.0354`.

    >>> import torch
    >>> from metatensor.torch import Labels, TensorBlock
    >>> from metatensor.torch.atomistic import System

    Define simple example structure

    >>> system = System(
    ...     types=torch.tensor([55, 17]),
    ...     positions=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    ...     cell=torch.eye(3),
    ... )

    Next we attach the charges to our ``system``

    >>> charges = torch.tensor([1.0, -1.0]).reshape(-1, 1)
    >>> data = TensorBlock(
    ...     values=charges,
    ...     samples=Labels.range("atom", charges.shape[0]),
    ...     components=[],
    ...     properties=Labels.range("charge", charges.shape[1]),
    ... )
    >>> system.add_data(name="charges", data=data)

    and compute the potenial

    >>> ewald = EwaldPotential()
    >>> potential = ewald.compute(system)

    The results are stored inside the ``values`` property inside the first
    :py:class:`TensorBlock <metatensor.torch.TensorBlock>` of the ``potential``.

    >>> potential[0].values
    tensor([[-2.0354],
            [ 2.0354]])

    Which is the same as the reference value given above.
    """

    def __init__(
        self,
        exponent: float = 1.0,
        sr_cutoff: Optional[torch.Tensor] = None,
        atomic_smearing: Optional[float] = None,
        lr_wavelength: Optional[float] = None,
        subtract_self: bool = True,
        subtract_interior: bool = False,
    ):
        _EwaldPotentialImpl.__init__(
            self,
            exponent=exponent,
            sr_cutoff=sr_cutoff,
            atomic_smearing=atomic_smearing,
            lr_wavelength=lr_wavelength,
            subtract_self=subtract_self,
            subtract_interior=subtract_interior,
        )
        CalculatorBaseMetatensor.__init__(self)
