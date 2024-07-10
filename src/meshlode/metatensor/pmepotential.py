from typing import Optional

import torch

from ..calculators.pmepotential import _PMEPotentialImpl
from .base import CalculatorBaseMetatensor


class PMEPotential(CalculatorBaseMetatensor, _PMEPotentialImpl):
    r"""Specie-wise long-range potential using a particle mesh-based Ewald (PME).

    Refer to :class:`meshlode.PMEPotential` for parameter documentation.

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
        sr_cutoff: Optional[torch.Tensor] = None,
        atomic_smearing: Optional[float] = None,
        mesh_spacing: Optional[float] = None,
        interpolation_order: int = 3,
        subtract_self: bool = True,
        subtract_interior: bool = False,
    ):
        _PMEPotentialImpl.__init__(
            self,
            exponent=exponent,
            sr_cutoff=sr_cutoff,
            atomic_smearing=atomic_smearing,
            mesh_spacing=mesh_spacing,
            interpolation_order=interpolation_order,
            subtract_self=subtract_self,
            subtract_interior=subtract_interior,
        )
        CalculatorBaseMetatensor.__init__(self)
