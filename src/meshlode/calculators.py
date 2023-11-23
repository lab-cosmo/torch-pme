"""
Available Calculators
=====================

Below is a list of all calculators available. Calculators are the core of MeshLODE and
are algorithms for transforming Cartesian coordinates into representations suitable for
machine learning.

Our calculator API follows the `rascaline <https://luthaf.fr/rascaline>`_ API and coding
guidelines to promote usability and interoperability with existing workflows.
"""
from typing import List, Optional, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from .system import System


class MeshPotential(torch.nn.Module):
    """A species wise long range potential.

    :param atomic_gaussian_width: Width of the atom-centered gaussian used to create the
        atomic density.
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis.
    :param interpolation_order: Interpolation order for mapping onto the grid.
        ``4`` equals cubic interpolation.

    Example
    -------

    >>> calculator = MeshPotential(atomic_gaussian_width=1)

    """

    name = "MeshPotential"

    def __init__(
        self,
        atomic_gaussian_width: float,
        mesh_spacing: float = 0.2,
        interpolation_order: float = 4,
    ):
        super().__init__()

        self.parameters = {
            "atomic_gaussian_width": atomic_gaussian_width,
            "mesh_spacing": mesh_spacing,
            "interpolation_order": interpolation_order,
        }

    def compute(
        self,
        systems: Union[System, List[System]],
        gradients: Optional[List[str]] = None,
    ) -> TensorMap:
        """Runs a calculation with this calculator on the given ``systems``.

        :param systems: single system or list of systems on which to run the
            calculation. If any of the systems' ``positions`` or ``cell`` has
            ``requires_grad`` set to :py:obj:`True`, then the corresponding gradients
            are computed and registered as a custom node in the computational graph, to
            allow backward propagation of the gradients later.
        :param gradients: List of forward gradients to keep in the output. If this is
            :py:obj:`None` or an empty list ``[]``, no gradients are kept in the output.
            Some gradients might still be computed at runtime to allow for backward
            propagation.
        """

        # Do actual calculations here...
        block = TensorBlock(
            samples=Labels.single(),
            components=[],
            properties=Labels.single(),
            values=torch.tensor([[1.0]]),
        )
        return TensorMap(keys=Labels.single(), blocks=[block])

    def forward(
        self,
        systems: List[System],
        gradients: Optional[List[str]] = None,
    ) -> TensorMap:
        """forward just calls :py:meth:`CalculatorModule.compute`"""
        return self.compute(systems=systems, gradients=gradients)
