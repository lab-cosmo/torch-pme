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

from meshlode.mesh_interpolator import MeshInterpolator
from meshlode.fourier_convolution import FourierSpaceConvolution
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
    
    def _compute_single_frame(self, cell: torch.tensor,
                              positions: torch.tensor, charges: torch.tensor,
                              subtract_self=False) -> torch.tensor:
        """
        Compute the "electrostatic" potential at the position of all atoms in a
        structure.

        :param cell: torch.tensor of shape (3,3). Describes the unit cell of the structure,
        where cell[i] is the i-th basis vector.
        :param positions: torch.tensor of shape (n_atoms, 3). Contains the Cartesian
        coordinates of the atoms. The implementation also works if the positions are
        not contained within the unit cell.
        :param charges: torch.tensor of shape (n_atoms, n_channels). In the simplest
        case, this would be a tensor of shape (n_atoms, 1) where charges[i,0] is the
        charge of atom i. More generally, the potential for the same atom positions
        is computed for n_channels independent meshes, and one can specify the "charge"
        of each atom on each of the meshes independently. For standard LODE that treats
        all atomic species separately, one example could be:
        If n_atoms = 4 and the species are [Na, Cl, Cl, Na], one could set n_channels=2
        and use the one-hot encoding
        charges = torch.tensor([[1,0],[0,1],[0,1],[1,0]])
        for the charges. This would then separately compute the "Na" potential and
        "Cl" potential. Subtracting these from each other, one could recover the more
        standard electrostatic potential in which Na and Cl have charges of +1 and -1,
        respectively.
        :param subtract_self: bool. If set to true, the contribution to the potential of
        the center atom itself is subtracted, meaning that only the potential generated
        by the remaining atoms + periodic images of the center atom is taken into
        account.

        :returns: torch.tensor of shape (n_atoms, n_channels) containing the potential
        at the position of each atom for the n_channels independent meshes separately.
        """
        smearing = self.parameters['atomic_gaussian_width']
        mesh_resolution = self.parameters['mesh_spacing']
        interpolation_order = self.parameters['interpolation_order']

        # Initializations
        n_atoms = len(positions)
        assert positions.shape == (n_atoms,3)
        assert charges.shape == (n_atoms, 1)

        # Define k-vectors
        if mesh_resolution is None:
            k_cutoff = 2 * torch.pi / (smearing / 2)
        else:
            k_cutoff = 2 * torch.pi / mesh_resolution

        # Compute number of times each basis vector of the
        # reciprocal space can be scaled until the cutoff
        # is reached
        basis_norms = torch.linalg.norm(cell, axis=1)
        ns_approx = k_cutoff * basis_norms / 2 / torch.pi
        ns_actual_approx = 2 * ns_approx + 1 # actual number of mesh points
        ns = 2**torch.ceil(torch.log2(ns_actual_approx)).long() # [nx, ny, nz]

        # Step 1: Smear particles onto mesh
        MI = MeshInterpolator(cell, ns, interpolation_order=interpolation_order)
        MI.compute_interpolation_weights(positions)
        rho_mesh = MI.points_to_mesh(particle_weights=charges)

        # Step 2: Perform Fourier space convolution (FSC)
        FSC = FourierSpaceConvolution(cell)
        kernel_func = lambda ksq: 4*torch.pi / ksq * torch.exp(-0.5*smearing**2*ksq)
        value_at_origin = 0. # charge neutrality
        potential_mesh = FSC.compute(rho_mesh, kernel_func, value_at_origin)

        # Step 3: Back interpolation
        interpolated_potential = MI.mesh_to_points(potential_mesh)

        # Remove self contribution
        if subtract_self:
            self_contrib = torch.sqrt(torch.tensor(2./torch.pi)) / smearing
            for i in range(n_atoms):
                interpolated_potential[i] -= charges[i,0] * self_contrib

        return interpolated_potential

