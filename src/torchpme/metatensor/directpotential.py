from ..calculators.directpotential import DirectPotential as DirectPotentialTorch
from .base import CalculatorBaseMetatensor


class DirectPotential(CalculatorBaseMetatensor):
    r"""
    Potential using a direct summation.

    Refer to :class:`torchpme.DirectPotential` for parameter documentation.

    Example
    -------
    We compute the energy of two charges which are sepearated by 2 along the z-axis.

    >>> import torch
    >>> from metatensor.torch import Labels, TensorBlock
    >>> from metatensor.torch.atomistic import System
    >>> from vesin.torch import NeighborList

    Define simple example structure

    >>> system = System(
    ...     types=torch.tensor([1, 1]),
    ...     positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
    ...     cell=torch.zeros([3, 3]),
    ... ).to(torch.float64)

    Next we attach the charges to our ``system``

    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> data = TensorBlock(
    ...     values=charges,
    ...     samples=Labels.range("atom", charges.shape[0]),
    ...     components=[],
    ...     properties=Labels.range("charge", charges.shape[1]),
    ... )
    >>> system.add_data(name="charges", data=data)

    We now compute the neighbor list using the ``vesin`` package. Refer to the
    `documentation <https://luthaf.fr/vesin>`_ for details on the API.

    >>> nl = NeighborList(cutoff=3.0, full_list=False)
    >>> i, j, S, D = nl.compute(
    ...     points=system.positions, box=system.cell, periodic=False, quantities="ijSD"
    ... )

    The ``vesin`` calculator returned the indices and the neighbor shifts. We know stack
    them together

    >>> neighbor_indices = torch.stack([i, j], dim=1)

    an attach the neighbor list to the above defined ``system`` object. For this we
    first create the ``samples`` metatadata for the :py:class:`TensorBlock
    <metatensor.torch.TensorBlock>` which will hold the neighbor list.

    >>> sample_values = torch.hstack([neighbor_indices, S])
    >>> samples = Labels(
    ...     names=[
    ...         "first_atom",
    ...         "second_atom",
    ...         "cell_shift_a",
    ...         "cell_shift_b",
    ...         "cell_shift_c",
    ...     ],
    ...     values=sample_values,
    ... )

    And wrap everything together.

    >>> values = D.reshape(-1, 3, 1)
    >>> neighbors = TensorBlock(
    ...     values=values,
    ...     samples=samples,
    ...     components=[Labels.range("xyz", 3)],
    ...     properties=Labels.range("distance", 1),
    ... )

    and compute the potenial. Note that you can optionally attach a neighbor list to the
    system to restrict the computation to certain pairs. See example of
    :py:class:`torchpme.metatensor.PMEPotential` for details on the process.

    >>> direct = DirectPotential()
    >>> potential = direct.forward(system, neighbors)

    The results are stored inside the ``values`` property inside the first
    :py:class:`TensorBlock <metatensor.torch.TensorBlock>` of the ``potential``.

    >>> potential[0].values
    tensor([[-0.2500],
            [ 0.2500]], dtype=torch.float64)

    Which is the expected potential since :math:`V \propto 1/r` where :math:`r` is the
    distance between the particles.

    """

    def __init__(self, exponent: float = 1.0, full_neighbor_list: bool = False):
        super().__init__()
        self.calculator = DirectPotentialTorch(
            exponent=exponent, full_neighbor_list=full_neighbor_list
        )
