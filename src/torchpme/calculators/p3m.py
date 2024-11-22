from typing import Optional

import torch
from torch import profiler

from ..lib.kspace_filter import P3MKSpaceFilter
from ..lib.kvectors import get_ns_mesh
from ..lib.mesh_interpolator import MeshInterpolator
from ..potentials import CoulombPotential
from .calculator import Calculator


class P3MCalculator(Calculator):
    r"""
    Potential using a particle-particle particle-mesh based Ewald (P3M).

    Scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N` used as a reference to test faster implementations.

    For getting reasonable values for the ``smaring`` of the potential class and  the
    ``mesh_spacing`` based on a given accuracy for a specific structure you should use
    :func:`torchpme.utils.tuning.tune_pme`. This function will also find the optimal
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
    :param diff_order: int, the order of the approximation of the difference operator.
        Higher order is more accurate, but also more expensive. For more details, see
        Appendix C of `that paper<http://dx.doi.org/10.1063/1.477414>`_. The values ``1,
        2, 3, 4, 5, 6`` are supported.
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.

    For an **example** on the usage for any calculator refer to :ref:`userdoc-how-to`.
    """

    def __init__(
        self,
        potential: CoulombPotential,
        mesh_spacing: float,
        interpolation_nodes: int = 4,
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
    ):
        self.mesh_spacing: float = mesh_spacing

        if interpolation_nodes not in [1, 2, 3, 4, 5]:
            raise ValueError("Only `interpolation_nodes` from 1 to 5 are allowed")
        self.interpolation_nodes: int = interpolation_nodes

        super().__init__(
            potential=potential,
            full_neighbor_list=full_neighbor_list,
            prefactor=prefactor,
        )

        # Initialize the filter module. Set dummy value for smearing to propper
        # initilize the `KSpaceFilter` below
        self.kspace_filter: P3MKSpaceFilter = P3MKSpaceFilter(
            cell=torch.eye(3),
            ns_mesh=torch.ones(3, dtype=int),
            interpolation_nodes=self.interpolation_nodes,
            kernel=self.potential,
            fft_norm="backward",
            ifft_norm="forward",
        )

        self.mesh_interpolator: MeshInterpolator = MeshInterpolator(
            cell=torch.eye(3),
            ns_mesh=torch.ones(3, dtype=int),
            interpolation_nodes=self.interpolation_nodes,
            method="P3M",
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

        with profiler.record_function("init 1: update mesh interpolator"):
            self.mesh_interpolator.update(cell, ns)

        with profiler.record_function("update the mesh for the k-space filter"):
            self.kspace_filter.update(cell, ns)

        with profiler.record_function("step 1: compute density interpolation"):
            self.mesh_interpolator.compute_weights(positions)
            rho_mesh = self.mesh_interpolator.points_to_mesh(particle_weights=charges)
            # print(rho_mesh)

        with profiler.record_function("step 2: perform actual convolution using FFT"):
            potential_mesh = self.kspace_filter.forward(rho_mesh)
            # print(potential_mesh)

        with profiler.record_function("step 3: back interpolation + volume scaling"):
            ivolume = torch.abs(cell.det()).pow(-1)
            interpolated_potential = (
                self.mesh_interpolator.mesh_to_points(potential_mesh) * ivolume
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
            prefac = self.potential.background_correction()
            interpolated_potential -= 2 * prefac * charge_tot * ivolume

        # Compensate for double counting of pairs (i,j) and (j,i)
        return interpolated_potential / 2


class _P3MCoulombPotential(CoulombPotential):
    r"""
    Coulomb potential for the P3M method.

    :param smearing: float or torch.Tensor containing the parameter often called "sigma"
        in publications, which determines the length-scale at which the short-range and
        long-range parts of the naive :math:`1/r` potential are separated. The smearing
        parameter corresponds to the "width" of a Gaussian smearing of the particle
        density.
    :param exclusion_radius: A length scale that defines a *local environment* within
        which the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param mode: int, 0 for the electrostatic potential, 1 for the electrostatic energy,
        2 for the dipolar torques, and 3 for the dipolar forces. For more details, see
        eq.30 of `this paper<https://doi.org/10.1063/1.3000389>`_
    :param diff_order: int, the order of the approximation of the difference operator.
        Higher order is more accurate, but also more expensive. For more details, see
        Appendix C of `that paper<http://dx.doi.org/10.1063/1.477414>`_. The values ``1,
        2, 3, 4, 5, 6`` are supported.
    :param dtype: type used for the internal buffers and parameters
    :param device: device used for the internal buffers and parameters
    """

    def __init__(
        self,
        mesh_spacing: float,
        interpolation_nodes: int,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        mode: int = 0,
        diff_order: int = 2,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)
        self.mesh_spacing = mesh_spacing
        self.interpolation_nodes = interpolation_nodes
        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"`mode` should be one of [0, 1, 2, 3], but got {mode}")
        self.mode = mode

        if diff_order not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"`diff_order` should be one of [1, 2, 3, 4, 5, 6], but got {diff_order}"
            )
        self.diff_order = diff_order

        # Dummy variables for initialization
        self._update_cell(torch.eye(3, device=device, dtype=dtype))
        self._actual_mesh_spacing: torch.Tensor = torch.tensor(1.0)

    def _update_cell(self, cell: torch.Tensor):
        self._cell = cell

    @torch.jit.export
    def kernel_from_kvectors(self, k: torch.Tensor) -> torch.Tensor:
        """
        Compatibility function with the interface of :py:class:`KSpaceKernel`, so that
        potentials can be used as kernels for :py:class:`KSpaceFilter`.
        """
        return self.lr_from_kvectors(k)

    def lr_from_kvectors(self, k: torch.Tensor) -> torch.Tensor:
        """
        Fourier transform of the LR part potential in terms of :math:`k`.

        :param k: torch.tensor containing the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range kernel without specifying `smearing`."
            )
        ONE_TENSOR = torch.tensor(1).to(device=k.device, dtype=k.dtype)
        cell_dimensions = torch.linalg.norm(self._cell, dim=1)
        self._actual_mesh_spacing = (
            cell_dimensions / get_ns_mesh(self._cell, self.mesh_spacing)
        ).reshape(1, 1, 1, 3)

        kh = k * self._actual_mesh_spacing.to(device=k.device, dtype=k.dtype)
        if self.mode == 0:
            # No need to calculate these things, all going to be one due to the zero exponent
            D = torch.ones(k.shape, dtype=k.dtype, device=k.device)
            D2 = torch.ones(k.shape[:3], dtype=k.dtype, device=k.device)
        else:
            D = self._diff_operator(kh)
            D2 = torch.linalg.norm(D, dim=-1) ** 2

        U2 = self._charge_assignment(kh)
        R = self._reference_force(k)

        # Calculate the kernel
        # See eq.30 of this paper https://doi.org/10.1063/1.3000389 for your main
        # reference, as well as the paragraph below eq.31.
        numerator = (
            U2
            * R
            * (ONE_TENSOR if self.mode == 0 else torch.sum(k * D, dim=-1) ** self.mode)
        )
        denominator = U2**2 * (ONE_TENSOR if self.mode == 0 else D2 ** (2 * self.mode))

        return torch.where(
            denominator == 0,
            0.0,
            numerator / denominator,
        )

    def _diff_operator(self, kh: torch.Tensor) -> torch.Tensor:
        """
        The approximation to the differential operator ``ik``. The ``i`` is taken
        out and cancels with the prefactor ``-i`` of the reference force function.
        See the Appendix C of this paper http://dx.doi.org/10.1063/1.477414.

        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, 3)
        """
        COEF: list[list[float]] = [  # coefficients for the difference operator
            [1.0],
            [4 / 3, -1 / 3],
            [3 / 2, -3 / 5, 1 / 10],
            [8 / 5, -4 / 5, 8 / 35, -1 / 35],
            [5 / 3, -20 / 21, 5 / 14, -5 / 63, 1 / 126],
            [12 / 7, -15 / 14, 10 / 21, -1 / 7, 2 / 77, -1 / 465],
        ]
        temp = torch.zeros(kh.shape, dtype=kh.dtype, device=kh.device)
        for i, coef in enumerate(COEF[self.diff_order - 1]):
            temp += (coef / (i + 1)) * torch.sin(kh * (i + 1))
        return temp / (self._actual_mesh_spacing)

    def _charge_assignment(self, kh: torch.Tensor) -> torch.Tensor:
        """
        The Fourier transformed charge assignment function divided by the volume of one mesh cell, in a squared form.
        See eq.18 and the paragraph below eq.31 of this paper
        http://dx.doi.org/10.1063/1.477414. Be aware that the volume cancels out
        with the prefactor of the assignment function (see eq.18).

        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd)
        """
        return torch.prod(
            torch.sinc(kh / (2 * torch.pi)),
            dim=-1,
        ) ** (self.interpolation_nodes * 2)

    def _reference_force(self, k: torch.Tensor) -> torch.Tensor:
        """
        The Fourier transform of the true reference force. See eq.32 of this paper
        http://dx.doi.org/10.1063/1.477414. In this implementation, the ``ik`` part
        is taken out and directly multiplied with the differential operator.

        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd)
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range kernel without specifying `smearing`."
            )

        k_sq = torch.linalg.norm(k, dim=-1) ** 2
        # avoid NaNs in backward, see
        # https://github.com/jax-ml/jax/issues/1052
        # https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
        masked = torch.where(k_sq == 0, 1.0, k_sq)
        return torch.where(
            k_sq == 0,
            0.0,
            4 * torch.pi * torch.exp(-0.5 * self.smearing**2 * masked) / masked,
        )


class _P3MSimpleCoulombPotential(CoulombPotential):
    r"""
    Coulomb potential for the P3M method.

    :param smearing: float or torch.Tensor containing the parameter often called "sigma"
        in publications, which determines the length-scale at which the short-range and
        long-range parts of the naive :math:`1/r` potential are separated. The smearing
        parameter corresponds to the "width" of a Gaussian smearing of the particle
        density.
    :param exclusion_radius: A length scale that defines a *local environment* within
        which the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param mode: int, 0 for the electrostatic potential, 1 for the electrostatic energy,
        2 for the dipolar torques, and 3 for the dipolar forces. For more details, see
        eq.30 of `this paper<https://doi.org/10.1063/1.3000389>`_
    :param diff_order: int, the order of the approximation of the difference operator.
        Higher order is more accurate, but also more expensive. For more details, see
        Appendix C of `that paper<http://dx.doi.org/10.1063/1.477414>`_. The values ``1,
        2, 3, 4, 5, 6`` are supported.
    :param dtype: type used for the internal buffers and parameters
    :param device: device used for the internal buffers and parameters
    """

    def __init__(
        self,
        mesh_spacing: float,
        interpolation_nodes: int,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        diff_order: int = 2,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(smearing, exclusion_radius, dtype, device)
        self._crude_mesh_spacing = mesh_spacing
        self.interpolation_nodes = interpolation_nodes
        if diff_order not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"`diff_order` should be one of [1, 2, 3, 4, 5, 6], but got {diff_order}"
            )
        self.diff_order = diff_order

        # Dummy variables for initialization
        self._update_cell(torch.eye(3))

    def _update_cell(self, cell: torch.Tensor):
        self._cell = cell

    @property
    def mesh_spacing(self) -> torch.Tensor:
        cell_dimensions = torch.linalg.norm(self._cell, dim=1)
        return (
            cell_dimensions / get_ns_mesh(self._cell, self._crude_mesh_spacing)
        ).reshape(1, 1, 1, 3)

    @torch.jit.export
    def kernel_from_kvectors(self, k: torch.Tensor) -> torch.Tensor:
        """
        Compatibility function with the interface of :py:class:`KSpaceKernel`, so that
        potentials can be used as kernels for :py:class:`KSpaceFilter`.
        """
        return self.lr_from_kvectors(k)

    def lr_from_kvectors(self, k: torch.Tensor) -> torch.Tensor:
        """
        Fourier transform of the LR part potential in terms of :math:`k`.

        :param k: torch.tensor containing the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        """
        kh = k * (self.mesh_spacing.to(device=k.device, dtype=k.dtype) / (2 * torch.pi))
        U2 = self._charge_assignment(kh)
        R = self._reference_force(k)

        # Calculate the kernel
        # See eq.30 of this paper https://doi.org/10.1063/1.3000389 for your main
        # reference, as well as the paragraph below eq.31.

        masked = torch.where(U2 == 0, 1.0, U2)
        return torch.where(
            U2 == 0,
            0.0,
            R / masked,
        )

    def _charge_assignment(self, kh: torch.Tensor) -> torch.Tensor:
        """
        The Fourier transformed charge assignment function divided by the volume of one mesh cell.
        See eq.18 and the paragraph below eq.31 of this paper
        http://dx.doi.org/10.1063/1.477414. Be aware that the volume cancels out
        with the prefactor of the assignment function (see eq.18).

        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd)
        """
        return torch.prod(
            torch.sinc(kh),
            dim=-1,
        ) ** (2 * self.interpolation_nodes)

    def _reference_force(self, k: torch.Tensor) -> torch.Tensor:
        """
        The Fourier transform of the true reference force. See eq.32 of this paper
        http://dx.doi.org/10.1063/1.477414. In this implementation, the ``ik`` part
        is taken out and directly multiplied with the differential operator.

        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd)
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range kernel without specifying `smearing`."
            )

        k_sq = torch.linalg.norm(k, dim=-1) ** 2
        # avoid NaNs in backward, see
        # https://github.com/jax-ml/jax/issues/1052
        # https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
        masked = torch.where(k_sq == 0, 1.0, k_sq)
        return torch.where(
            k_sq == 0,
            0.0,
            4 * torch.pi * torch.exp(-0.5 * self.smearing**2 * masked) / masked,
        )
