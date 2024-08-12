from typing import Optional

from ..calculators.pmepotential import _PMEPotentialImpl
from .base import CalculatorBaseMetatensor


class PMEPotential(CalculatorBaseMetatensor, _PMEPotentialImpl):
    r"""Specie-wise long-range potential using a particle mesh-based Ewald (PME).

    Refer to :class:`torchpme.PMEPotential` for parameter documentation.

    Example
    -------
    We calculate the Madelung constant of a CsCl (Cesium-Chloride) crystal. The
    reference value is :math:`2 \cdot 1.7626 / \sqrt{3} \approx 2.0354`.

    >>> import torch
    >>> from metatensor.torch import Labels, TensorBlock
    >>> from metatensor.torch.atomistic import System, NeighborListOptions
    >>> from vesin import NeighborList

    Define simple example structure

    >>> system = System(
    ...     types=torch.tensor([55, 17]),
    ...     positions=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    ...     cell=torch.eye(3),
    ... )

    Next, we attach the charges to our ``system``

    >>> charges = torch.tensor([1.0, -1.0]).reshape(-1, 1)
    >>> data = TensorBlock(
    ...     values=charges,
    ...     samples=Labels.range("atom", charges.shape[0]),
    ...     components=[],
    ...     properties=Labels.range("charge", charges.shape[1]),
    ... )
    >>> system.add_data(name="charges", data=data)

    Compute the neighbor indices (``"i"``, ``"j"``) and the neighbor shifts ("``S``")
    using the ``vesin`` package. Refer to the `documentation
    <https://luthaf.fr/vesin>`_ for details on the API. Similarly you can also use
    ``ase``'s :py:func:`neighbor_list <ase.neighborlist.neighbor_list>`.

    >>> cell_dimensions = torch.linalg.norm(system.cell, dim=1)
    >>> cutoff = torch.min(cell_dimensions) / 2 - 1e-6
    >>> nl = NeighborList(cutoff=cutoff, full_list=True)
    >>> i, j, S, D = nl.compute(
    ...     points=system.positions, box=system.cell, periodic=True, quantities="ijSD"
    ... )

    The ``vesin`` calculator returned the indices and the neighbor shifts. We know stack
    the together and convert them into the suitable types

    >>> i = torch.from_numpy(i.astype(int))
    >>> j = torch.from_numpy(j.astype(int))
    >>> neighbor_indices = torch.vstack([i, j])
    >>> neighbor_shifts = torch.from_numpy(S.astype(int))

    If you inspect the neighborlist you will notice that they are empty for the given
    system, which means the the whole potential will be calculated using the long range
    part of the potential.

    We now attach the neighbor list to the above defined ``system`` object. For this we
    first create the ``samples`` metatadata for the :py:class:`TensorBlock
    <metatensor.torch.TensorBlock>` which will hold the neighbor list.

    >>> sample_values = torch.hstack([neighbor_indices.T, neighbor_shifts])
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

    And wrap everything together and add it to our ``system``.

    >>> values = torch.from_numpy(D).reshape(-1, 3, 1)
    >>> values = values.type(system.positions.dtype)
    >>> neighbors = TensorBlock(
    ...     values=values,
    ...     samples=samples,
    ...     components=[Labels.range("xyz", 3)],
    ...     properties=Labels.range("distance", 1),
    ... )
    >>> nl_options = NeighborListOptions(cutoff=cutoff, full_list=True)
    >>> system.add_neighbor_list(options=nl_options, neighbors=neighbors)


    Finally, we initlize the potential class and ``compute`` the
    potential for the crystal

    >>> pme = PMEPotential()
    >>> potential = pme.compute(system)

    The results are stored inside the ``values`` property inside the first
    :py:class:`TensorBlock <metatensor.torch.TensorBlock>` of the ``potential``.

    >>> potential[0].values
    tensor([[-2.0384],
            [ 2.0384]])

    Which is the same as the reference value given above.
    """

    def __init__(
        self,
        exponent: float = 1.0,
        atomic_smearing: Optional[float] = None,
        mesh_spacing: Optional[float] = None,
        interpolation_order: int = 3,
        subtract_self: bool = True,
        subtract_interior: bool = False,
    ):
        _PMEPotentialImpl.__init__(
            self,
            exponent=exponent,
            atomic_smearing=atomic_smearing,
            mesh_spacing=mesh_spacing,
            interpolation_order=interpolation_order,
            subtract_self=subtract_self,
            subtract_interior=subtract_interior,
        )
        CalculatorBaseMetatensor.__init__(self)
