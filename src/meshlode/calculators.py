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


class MeshLodeSphericalExpansion(torch.nn.Module):
    """Mesh Long-Distance Equivariant (LODE).

    :param cutoff: Spherical real space cutoff to use for atomic environments. Note that
        this cutoff is only used for the projection of the density. In contrast to SOAP,
        LODE also takes atoms outside of this cutoff into account for the density.
    :param max_radial: Number of radial basis function to use in the expansion
    :param max_angular: Number of spherical harmonics to use in the expansion
    :param atomic_gaussian_width: Width of the atom-centered gaussian used to create the
        atomic density.
    :param center_atom_weight: Weight of the central atom contribution in the central
        image to the features. If `1` the center atom contribution is weighted the same
        as any other contribution. If `0` the central atom does not contribute to the
        features at all.
    :param radial_basis: Radial basis to use for the radial integral
    :param potential_exponent: Potential exponent of the decorated atom density.

    Example
    -------

    >>> calculator = MeshLodeSphericalExpansion(
    ...     cutoff=2.0,
    ...     max_radial=8,
    ...     max_angular=6,
    ...     atomic_gaussian_width=1,
    ...     radial_basis={"Gto"},
    ...     potential_exponent=1,
    ... )


    """

    name = "MeshLodeSphericalExpansion"

    def __init__(
        self,
        cutoff: float,
        max_radial: int,
        max_angular: int,
        atomic_gaussian_width: float,
        potential_exponent: int,
        radial_basis: dict,
    ):
        super().__init__()

        self.parameters = {
            "cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "atomic_gaussian_width": atomic_gaussian_width,
            "potential_exponent": potential_exponent,
            "radial_basis": radial_basis,
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
