import torch

from ..lib.kspace_filter import P3MKSpaceFilter
from ..lib.mesh_interpolator import MeshInterpolator
from ..potentials import Potential
from .pme import PMECalculator


class P3MCalculator(PMECalculator):
    r"""
    Potential using a particle-particle particle-mesh based Ewald (P3M).

    For getting reasonable values for the ``smearing`` of the potential class and  the
    ``mesh_spacing`` based on a given accuracy for a specific structure you should use
    :func:`torchpme.utils.tuning.tune_p3m`. This function will also find the optimal
    ``cutoff`` for the  **neighborlist**.

    .. hint::

        For a training exercise it is recommended only run a tuning procedure with
        :func:`torchpme.utils.tuning.tune_p3m` for the largest system in your dataset.

    :param potential: A :py:class:`Potential` object that implements the evaluation
        of short and long-range potential terms. The ``smearing`` parameter
        of the potential determines the split between real and k-space regions.
        For a :py:class:`torchpme.lib.CoulombPotential` it corresponds
        to the smearing of the atom-centered Gaussian used to split the
        Coulomb potential into the short- and long-range parts. A reasonable value for
        most systems is to set it to ``1/5`` times the neighbor list cutoff.
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically be set
        to half of ``smearing``.
    :param interpolation_nodes: The number ``n`` of nodes used in the interpolation per
        coordinate axis. The total number of interpolation nodes in 3D will be ``n^3``.
        In general, for ``n`` nodes, the interpolation will be performed by piecewise
        polynomials of degree ``n - 1`` (e.g. ``n = 4`` for cubic interpolation).
        Only the values ``1, 2, 3, 4, 5`` are supported.
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.

    For an **example** on the usage for any calculator refer to :ref:`userdoc-how-to`.
    """

    def __init__(
        self,
        potential: Potential,
        mesh_spacing: float,
        interpolation_nodes: int = 4,
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
    ):
        self.mesh_spacing: float = mesh_spacing

        if interpolation_nodes not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_nodes` from 1 to 5 are allowed")
        self.interpolation_nodes: int = interpolation_nodes

        super(PMECalculator, self).__init__(
            potential=potential,
            full_neighbor_list=full_neighbor_list,
            prefactor=prefactor,
        )

        self.kspace_filter: P3MKSpaceFilter = P3MKSpaceFilter(
            cell=torch.eye(3),
            ns_mesh=torch.ones(3, dtype=int),
            interpolation_nodes=self.interpolation_nodes,
            kernel=self.potential,
            mode=0,  # Green's function for point-charge potentials
            differential_order=2,  # Order of the discretization of the differential operator
            fft_norm="backward",
            ifft_norm="forward",
        )

        self.mesh_interpolator: MeshInterpolator = MeshInterpolator(
            cell=torch.eye(3),
            ns_mesh=torch.ones(3, dtype=int),
            interpolation_nodes=self.interpolation_nodes,
            method="P3M",
        )
