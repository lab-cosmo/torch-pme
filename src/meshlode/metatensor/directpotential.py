from ..calculators.directpotential import _DirectPotentialImpl
from .base import CalculatorBaseMetatensor


class DirectPotential(CalculatorBaseMetatensor, _DirectPotentialImpl):
    r"""Specie-wise long-range potential using a direct summation over all atoms.

    Refer to :class:`meshlode.DirectPotential` for parameter documentation.

    Example
    -------
    We compute the energy of two charges which are sepearated by 2 along the z-axis.

    >>> import torch
    >>> from metatensor.torch import Labels, TensorBlock
    >>> from metatensor.torch.atomistic import System

    Define simple example structure

    >>> system = System(
    ...     types=torch.tensor([1, 1]),
    ...     positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
    ...     cell=torch.zeros([3, 3]),
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

    >>> direct = DirectPotential()
    >>> potential = direct.compute(system)

    The results are stored inside the ``values`` property inside the first
    :py:class:`TensorBlock <metatensor.torch.TensorBlock>` of the ``potential``.

    >>> potential[0].values
    tensor([[-0.5000],
            [ 0.5000]])

    Which is the expected potential since :math:`V \propto 1/r` where :math:`r` is the
    distance between the particles.
    """

    def __init__(self, exponent: float = 1.0):
        _DirectPotentialImpl.__init__(self, exponent=exponent)
        CalculatorBaseMetatensor.__init__(self, exponent=exponent)
