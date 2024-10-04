from typing import Optional

import torch
from torch import profiler

from ..lib import Potential
from ..lib.kspace_filter import KSpaceFilter
from ..lib.kvectors import get_ns_mesh
from ..lib.mesh_interpolator import MeshInterpolator
from .calculatorbase import Calculator


class PMECalculator(Calculator):
    r"""
    Potential using a particle mesh-based Ewald (PME).

    Scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N` used as a reference to test faster implementations.

    For computing a **neighborlist** a reasonable ``cutoff`` is half the length of
    the shortest cell vector, which can be for example computed according as

    .. code-block:: python

        cell_dimensions = torch.linalg.norm(cell, dim=1)
        cutoff = torch.min(cell_dimensions) / 2 - 1e-6

    This ensures a accuracy of the short range part of ``1e-5``.

    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param atomic_smearing: Width of the atom-centered Gaussian used to split the
        Coulomb potential into the short- and long-range parts. A reasonable value for
        most systems is to set it to ``1/5`` times the neighbor list cutoff. If
        :py:obj:`None` ,it will be set to 1/5 times of half the largest box vector
        (separately for each structure).
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically be set
        to half of ``atomic_smearing``.
    :param interpolation_order: Interpolation order for mapping onto the grid, where an
        interpolation order of p corresponds to interpolation by a polynomial of degree
        ``p - 1`` (e.g. ``p = 4`` for cubic interpolation).
    :param subtract_interior: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from all atoms within the cutoff
        Note that if set to true, the self contribution (see previous) is also
        subtracted by default.

    Example
    -------
    We calculate the Madelung constant of a CsCl (Cesium-Chloride) crystal. The
    reference value is :math:`2 \cdot 1.7626 / \sqrt{3} \approx 2.0354`.

    >>> import torch
    >>> from vesin.torch import NeighborList

    Define crystal structure

    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)

    Compute the neighbor indices (``"i"``, ``"j"``) and the neighbor distances ("``d``")
    using the ``vesin`` package. Refer to the `documentation <https://luthaf.fr/vesin>`_
    for details on the API.

    >>> cell_dimensions = torch.linalg.norm(cell, dim=1)
    >>> cutoff = torch.min(cell_dimensions) / 2 - 1e-6
    >>> nl = NeighborList(cutoff=cutoff, full_list=False)
    >>> i, j, neighbor_distances = nl.compute(
    ...     points=positions, box=cell, periodic=True, quantities="ijd"
    ... )
    >>> neighbor_indices = torch.stack([i, j], dim=1)

    If you inspect the neighbor list you will notice that the tensors are empty for the
    given system, which means the the whole potential will be calculated using the long
    range part of the potential. Finally, we initlize the potential class and
    ``compute`` the potential for the crystal.

    >>> pme = CalculatorPME()
    >>> pme.forward(
    ...     charges=charges,
    ...     cell=cell,
    ...     positions=positions,
    ...     neighbor_indices=neighbor_indices,
    ...     neighbor_distances=neighbor_distances,
    ... )
    tensor([[-1.0192],
            [ 1.0192]], dtype=torch.float64)

    Which is close to the reference value given above.

    """

    def __init__(
        self,
        potential: Potential,
        mesh_spacing: Optional[float] = None,
        interpolation_order: int = 3,
        full_neighbor_list: bool = False,
    ):
        super().__init__(
            potential=potential,
            full_neighbor_list=full_neighbor_list,
        )

        if mesh_spacing is None:
            mesh_spacing = potential.range_radius / 8.0
        self.mesh_spacing = mesh_spacing

        if interpolation_order not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_order` from 1 to 5 are allowed")
        self.interpolation_order = interpolation_order

        # Initialize the filter module. Set dummy value for smearing to propper
        # initilize the `KSpaceFilter` below
        self._KF = KSpaceFilter(
            cell=torch.eye(3),
            ns_mesh=torch.ones(3, dtype=int),
            kernel=self.potential,
            fft_norm="backward",
            ifft_norm="forward",
        )

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: Kernel function `G` and initialization of `MeshInterpolator` only depend
        # on `cell`. Caching may save up to 15% but issues with AD should be taken
        # resolved.

        with profiler.record_function("init 0: preparation"):
            # Compute number of times each basis vector of the reciprocal space can be
            # scaled until the cutoff is reached
            ns = get_ns_mesh(cell, self.mesh_spacing)

        with profiler.record_function("init 1: initialize mesh interpolator"):
            interpolator = MeshInterpolator(cell, ns, order=self.interpolation_order)

        with profiler.record_function("update the mesh for the k-space filter"):
            self._KF.update_mesh(cell, ns)

        with profiler.record_function("step 1: compute density interpolation"):
            interpolator.compute_weights(positions)
            rho_mesh = interpolator.points_to_mesh(particle_weights=charges)

        with profiler.record_function("step 2: perform actual convolution using FFT"):
            potential_mesh = self._KF.compute(rho_mesh)

        with profiler.record_function("step 3: back interpolation + volume scaling"):
            ivolume = torch.abs(cell.det()).pow(-1)
            interpolated_potential = (
                interpolator.mesh_to_points(potential_mesh) * ivolume
            )

        with profiler.record_function("step 4: remove the self-contribution"):
            # Using the Coulomb potential as an example, this is the potential generated
            # at the origin by the fictituous Gaussian charge density in order to split
            # the potential into a SR and LR part. This contribution always should be
            # subtracted since it depends on the smearing parameter, which is purely a
            # convergence parameter.
            fill_value = self.potential.self_contribution()
            self_contrib = torch.full([], fill_value, device=self._device)
            interpolated_potential -= charges * self_contrib

        with profiler.record_function("step 5: charge neutralization"):
            # If the cell has a net charge (i.e. if sum(charges) != 0), the method
            # implicitly assumes that a homogeneous background charge of the opposite
            # sign is present to make the cell neutral. In this case, the potential has
            # to be adjusted to compensate for this. An extra factor of 2 is added to
            # compensate for the division by 2 later on
            charge_tot = torch.sum(charges, dim=0)
            prefac = self.potential.charge_correction()
            interpolated_potential -= 2 * prefac * charge_tot * ivolume

        # Compensate for double counting of pairs (i,j) and (j,i)
        return interpolated_potential / 2


# TODO Remove below this line
from .calculatorbase import CalculatorBaseTorch


class CalculatorPME(CalculatorBaseTorch):
    r"""
    Potential using a particle mesh-based Ewald (PME).

    Scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N` used as a reference to test faster implementations.

    For computing a **neighborlist** a reasonable ``cutoff`` is half the length of
    the shortest cell vector, which can be for example computed according as

    .. code-block:: python

        cell_dimensions = torch.linalg.norm(cell, dim=1)
        cutoff = torch.min(cell_dimensions) / 2 - 1e-6

    This ensures a accuracy of the short range part of ``1e-5``.

    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param atomic_smearing: Width of the atom-centered Gaussian used to split the
        Coulomb potential into the short- and long-range parts. A reasonable value for
        most systems is to set it to ``1/5`` times the neighbor list cutoff. If
        :py:obj:`None` ,it will be set to 1/5 times of half the largest box vector
        (separately for each structure).
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically be set
        to half of ``atomic_smearing``.
    :param interpolation_order: Interpolation order for mapping onto the grid, where an
        interpolation order of p corresponds to interpolation by a polynomial of degree
        ``p - 1`` (e.g. ``p = 4`` for cubic interpolation).
    :param subtract_interior: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from all atoms within the cutoff
        Note that if set to true, the self contribution (see previous) is also
        subtracted by default.

    Example
    -------
    We calculate the Madelung constant of a CsCl (Cesium-Chloride) crystal. The
    reference value is :math:`2 \cdot 1.7626 / \sqrt{3} \approx 2.0354`.

    >>> import torch
    >>> from vesin.torch import NeighborList

    Define crystal structure

    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)

    Compute the neighbor indices (``"i"``, ``"j"``) and the neighbor distances ("``d``")
    using the ``vesin`` package. Refer to the `documentation <https://luthaf.fr/vesin>`_
    for details on the API.

    >>> cell_dimensions = torch.linalg.norm(cell, dim=1)
    >>> cutoff = torch.min(cell_dimensions) / 2 - 1e-6
    >>> nl = NeighborList(cutoff=cutoff, full_list=False)
    >>> i, j, neighbor_distances = nl.compute(
    ...     points=positions, box=cell, periodic=True, quantities="ijd"
    ... )
    >>> neighbor_indices = torch.stack([i, j], dim=1)

    If you inspect the neighbor list you will notice that the tensors are empty for the
    given system, which means the the whole potential will be calculated using the long
    range part of the potential. Finally, we initlize the potential class and
    ``compute`` the potential for the crystal.

    >>> pme = CalculatorPME()
    >>> pme.forward(
    ...     positions=positions,
    ...     charges=charges,
    ...     cell=cell,
    ...     neighbor_indices=neighbor_indices,
    ...     neighbor_distances=neighbor_distances,
    ... )
    tensor([[-1.0192],
            [ 1.0192]], dtype=torch.float64)

    Which is close to the reference value given above.

    """

    def __init__(
        self,
        potential: Potential,
        mesh_spacing: Optional[float] = None,
        interpolation_order: int = 3,
        subtract_interior: bool = False,
        full_neighbor_list: bool = False,
    ):
        super().__init__(
            potential=potential,
            full_neighbor_list=full_neighbor_list,
        )

        self.mesh_spacing = mesh_spacing
        self.subtract_interior = subtract_interior

        if interpolation_order not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_order` from 1 to 5 are allowed")
        self.interpolation_order = interpolation_order

        # Initialize the filter module. Set dummy value for smearing to propper
        # initilize the `KSpaceFilter` below
        self._KF = KSpaceFilter(
            cell=torch.eye(3),
            ns_mesh=torch.ones(3, dtype=int),
            kernel=self.potential,
            fft_norm="backward",
            ifft_norm="forward",
        )

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        # Set the defaut values of convergence parameters The total computational cost =
        # cost of SR part + cost of LR part Bigger smearing increases the cost of the SR
        # part while decreasing the cost of the LR part. Since the latter usually is
        # more expensive, we maximize the value of the smearing by default to minimize
        # the cost of the LR part. The auxilary parameter mesh_spacing then controls the
        # convergence of the SR and LR sums, respectively. The default values are chosen
        # to reach a convergence on the order of 1e-4 to 1e-5 for the test structures.
        if self.mesh_spacing is None:
            mesh_spacing = self.potential.range_radius / 8.0
        else:
            mesh_spacing = self.mesh_spacing

        with profiler.record_function("compute SR"):
            # use real space sum
            potential_sr = self._compute_sr(
                is_periodic=True,
                charges=charges,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
            )

        with profiler.record_function("compute LR"):
            # use Fourier / reciprocal space sum
            potential_lr = self._compute_lr(
                positions=positions,
                charges=charges,
                cell=cell,
                lr_wavelength=mesh_spacing,
            )

        return potential_sr + potential_lr

    def _compute_lr(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        lr_wavelength: float,
    ) -> torch.Tensor:
        # TODO: Kernel function `G` and initialization of `MeshInterpolator` only depend
        # on `cell`. Caching may save up to 15% but issues with AD should be taken
        # resolved.

        with profiler.record_function("init 0: preparation"):
            # Compute number of times each basis vector of the reciprocal space can be
            # scaled until the cutoff is reached
            ns = get_ns_mesh(cell, lr_wavelength)

        with profiler.record_function("init 1: initialize mesh interpolator"):
            interpolator = MeshInterpolator(cell, ns, order=self.interpolation_order)

        with profiler.record_function("update the mesh for the k-space filter"):
            self._KF.update_mesh(cell, ns)

        with profiler.record_function("step 1: compute density interpolation"):
            interpolator.compute_weights(positions)
            rho_mesh = interpolator.points_to_mesh(particle_weights=charges)

        with profiler.record_function("step 2: perform actual convolution using FFT"):
            potential_mesh = self._KF.compute(rho_mesh)

        with profiler.record_function("step 3: back interpolation + volume scaling"):
            ivolume = torch.abs(cell.det()).pow(-1)
            interpolated_potential = (
                interpolator.mesh_to_points(potential_mesh) * ivolume
            )

        with profiler.record_function("step 4: remove the self-contribution"):
            # Using the Coulomb potential as an example, this is the potential generated
            # at the origin by the fictituous Gaussian charge density in order to split
            # the potential into a SR and LR part. This contribution always should be
            # subtracted since it depends on the smearing parameter, which is purely a
            # convergence parameter.
            fill_value = self.potential.self_contribution
            self_contrib = torch.full([], fill_value, device=self._device)
            interpolated_potential -= charges * self_contrib

        with profiler.record_function("step 5: charge neutralization"):
            # If the cell has a net charge (i.e. if sum(charges) != 0), the method
            # implicitly assumes that a homogeneous background charge of the opposite
            # sign is present to make the cell neutral. In this case, the potential has
            # to be adjusted to compensate for this. An extra factor of 2 is added to
            # compensate for the division by 2 later on
            charge_tot = torch.sum(charges, dim=0)
            prefac = self.potential.charge_correction_prefac
            interpolated_potential -= 2 * prefac * charge_tot * ivolume

        # Compensate for double counting of pairs (i,j) and (j,i)
        return interpolated_potential / 2
