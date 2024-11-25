from typing import Optional

import torch

# from ..potentials import Potential
from .kvectors import generate_kvectors_for_mesh


class KSpaceKernel(torch.nn.Module):
    r"""
    Base class defining the interface for a reciprocal-space kernel helper.

    Provides an interface to compute the reciprocal-space convolution kernel
    that is used e.g. to compute potentials using Fourier transforms. Parameters
    of the kernel in derived classes should be defined and stored in the
    ``__init__`` method.

    NB: we need this slightly convoluted way of implementing what often amounts
    to a simple, pure function of :math:`|\mathbf{k}|^2` in order to be able to
    provide a customizable filter class that can be jitted.
    """

    def __init__(self):
        super().__init__()

    def kernel_from_k_sq(self, kvectors: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the reciprocal-space kernel on a grid of k points given a tensor
        containing :math:`\mathbf{k}`.

        :param kvectors: torch.tensor containing the k vector at which the
            kernel is to be evaluated.
        """
        raise NotImplementedError(
            f"kernel_from_k_sq is not implemented for '{self.__class__.__name__}'"
        )


class KSpaceFilter(torch.nn.Module):
    r"""
    Apply a reciprocal-space filter to a real-space mesh.

    The class combines the costruction of a reciprocal-space grid
    :math:`\{\mathbf{k}_n\}`
    (that should be commensurate to the grid in real space, so the class takes
    the same options as :class:`MeshInterpolator`), the calculation of
    a scalar filter function :math:`\phi(|\mathbf{k}|^2)`, defined as a function of
    the squared norm of the reciprocal space grid points, and the application
    of the filter to a real-space function :math:`f(\mathbf{x})`,
    defined on a mesh :math:`\{\mathbf{x}_n\}`.

    In practice, the application of the filter amounts to
    :math:`f\rightarrow \hat{f} \rightarrow \hat{\tilde{f}}=
    \hat{f} \phi \rightarrow \tilde{f}`

    See also the :ref:`example-kspace-demo` for a demonstration of the functionalities
    of this class.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param ns_mesh: toch.tensor of shape ``(3,)``
        Number of mesh points to use along each of the three axes
    :param kernel: KSpaceKernel
        A KSpaceKernel-derived class providing a ``from_k_sq`` method that
        evaluates :math:`\psi` given the square modulus of
        the k-space mesh points
    :param fft_norm: str
        The normalization applied to the forward FT. Can be
        "forward", "backward", "ortho". See :func:`torch:fft:rfftn`
    :param ifft_norm: str
        The normalization applied to the inverse FT. Can be
        "forward", "backward", "ortho". See :func:`torch:fft:irfftn`
    """

    def __init__(
        self,
        cell: torch.Tensor,
        ns_mesh: torch.Tensor,
        kernel: KSpaceKernel,
        fft_norm: str = "ortho",
        ifft_norm: str = "ortho",
    ):
        super().__init__()

        self._fft_norm = fft_norm
        self._ifft_norm = ifft_norm
        if fft_norm not in ["ortho", "forward", "backward"]:
            raise ValueError(
                f"Invalid option '{fft_norm}' for the `fft_norm` parameter."
            )
        if ifft_norm not in ["ortho", "forward", "backward"]:
            raise ValueError(
                f"Invalid option '{ifft_norm}' for the `ifft_norm` parameter."
            )

        self.kernel = kernel
        self.update(cell, ns_mesh)

    @torch.jit.export
    def update(
        self,
        cell: Optional[torch.Tensor] = None,
        ns_mesh: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update buffers and derived attributes of the instance.

        If neither ``cell`` nor ``ns_mesh`` are passed, only the filter is updated,
        typically following a change in the underlying potential. If ``cell`` and/or
        ``ns_mesh`` are given, the instance's attributes required by these will also be
        updated accordingly.

        :param cell: torch.tensor of shape ``(3, 3)``, where `
            `cell[i]`` is the i-th basis vector of the unit cell
        :param ns_mesh: toch.tensor of shape ``(3,)``
            Number of mesh points to use along each of the three axes
        """
        self._prep_kvectors(cell, ns_mesh)

        # always update the kfilter to reduce the risk it'd go out of sync if the is an
        # update in the underlaying potential
        self._kfilter = self.kernel.kernel_from_k_sq(self._k_sq)

    def forward(self, mesh_values: torch.Tensor) -> torch.Tensor:
        """
        Applies the k-space filter by Fourier transforming the given
        ``mesh_values`` tensor, multiplying the result by the filter array
        (that should have been previously computed with a call to
        :func:`update`) and Fourier-transforming back
        to real space.

        If you update the ``cell``, the ``ns_mesh`` or anything inside the ``kernel``
        object after you initlized the object, you have call :meth:`update` to update
        the object calling this method.

        .. code-block:: python

            kernel_filter.update(cell)
            kernel_filter.forward(mesh)

        :param mesh_values: torch.tensor of shape ``(n_channels, nx, ny, nz)``
            The values of the input function on a real-space mesh. Shape
            should match the shape of the filter.

        :returns: torch.tensor of shape ``(n_channels, nx, ny, nz)``
            The real-space mesh containing the transformed function values.
        """
        if mesh_values.dim() != 4:
            raise ValueError(
                "`mesh_values` needs to be a 4 dimensional tensor, got "
                f"{mesh_values.dim()}"
            )

        if mesh_values.device != self._kfilter.device:
            raise ValueError(
                "`mesh_values` and the k-space filter are on different devices, got "
                f"{mesh_values.device} and {self._kfilter.device}"
            )

        # Applying the Fourier filter involves the following substeps:
        # 1. Fourier transform the input mesh
        # 2. multiply by kernel in k-space
        # 3. transform back
        # For the Fourier transforms, we use the normalization conditions
        # that do not introduce any extra factors of 1/n_mesh.
        # This is why the forward transform (fft) is called with the
        # normalization option 'backward' (the convention in which 1/n_mesh
        # is in the backward transformation) and vice versa for the
        # inverse transform (irfft).

        dims = (1, 2, 3)  # dimensions along which to Fourier transform
        mesh_hat = torch.fft.rfftn(mesh_values, norm=self._fft_norm, dim=dims)

        if mesh_hat.shape[-3:] != self._kfilter.shape[-3:]:
            raise ValueError(
                "The real-space mesh is inconsistent with the k-space grid."
            )

        filter_hat = mesh_hat * self._kfilter

        return torch.fft.irfftn(
            filter_hat,
            norm=self._ifft_norm,
            dim=dims,
            # NB: we must specify the size of the output
            # as for certain mesh sizes the inverse FT is not
            # well-defined
            s=mesh_values.shape[-3:],
        )

    def _prep_kvectors(
        self, cell: Optional[torch.Tensor], ns_mesh: Optional[torch.Tensor]
    ):
        if cell is not None:
            if cell.shape != (3, 3):
                raise ValueError(
                    f"cell of shape {list(cell.shape)} should be of shape (3, 3)"
                )
            self.cell = cell

        if ns_mesh is not None:
            if ns_mesh.shape != (3,):
                raise ValueError(
                    f"shape {list(ns_mesh.shape)} of `ns_mesh` has to be (3,)"
                )
            self.ns_mesh = ns_mesh

        if self.cell.device != self.ns_mesh.device:
            raise ValueError(
                "`cell` and `ns_mesh` are on different devices, got "
                f"{self.cell.device} and {self.ns_mesh.device}"
            )

        if cell is not None or ns_mesh is not None:
            self._kvectors = generate_kvectors_for_mesh(ns=self.ns_mesh, cell=self.cell)
            self._k_sq = torch.linalg.norm(self._kvectors, dim=3) ** 2


class P3MKSpaceFilter(KSpaceFilter):
    r"""
    A specialized implementation of the k-space filter for the P3M method, with a
    cell-dependent Green's function kernel. This class does almost the same thing as
    :class:`KSpaceFilter`, but with a different, P3M-specialized filter. See `this paper
    <http://dx.doi.org/10.1063/1.477414>`_ for your reference.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param ns_mesh: toch.tensor of shape ``(3,)`` Number of mesh points to use along
        each of the three axes
    :param interpolation_nodes: int The number ``n`` of nodes used in the interpolation
        per coordinate axis. The total number of interpolation nodes in 3D will be
        ``n^3``. In general, for ``n`` nodes, the interpolation will be performed by
        piecewise polynomials of degree ``n - 1`` (e.g. ``n = 4`` for cubic
        interpolation). Only the values ``1, 2, 3, 4, 5`` are supported.
    :param kernel: KSpaceKernel A KSpaceKernel-derived class providing a ``from_k_sq``
        method that evaluates :math:`\psi` given the square modulus of the k-space mesh
        points
    :param fft_norm: str The normalization applied to the forward FT. Can be "forward",
        "backward", "ortho". See :func:`torch:fft:rfftn`
    :param ifft_norm: str The normalization applied to the inverse FT. Can be "forward",
        "backward", "ortho". See :func:`torch:fft:irfftn`
    :param mode: int, 0 for the electrostatic potential, 1 for the electrostatic energy,
        2 for the dipolar torques, and 3 for the dipolar forces. For more details, see
        eq.30 of `that paper <https://doi.org/10.1063/1.3000389>`_.
    :param diff_order: int, the order of the approximation of the difference operator.
        Higher order is more accurate, but also more expensive. For more details, see
        Appendix C of `this paper <http://dx.doi.org/10.1063/1.477414>`_. The values
        ``1, 2, 3, 4, 5, 6`` are supported.
    """

    def __init__(
        self,
        cell: torch.Tensor,
        ns_mesh: torch.Tensor,
        interpolation_nodes: int,
        kernel: KSpaceKernel,
        fft_norm: str = "ortho",
        ifft_norm: str = "ortho",
        mode: int = 0,
        differential_order: int = 2,
    ):
        self.interpolation_nodes = interpolation_nodes
        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"`mode` should be one of [0, 1, 2, 3], but got {mode}")
        self.mode = mode
        if differential_order not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"`differential_order` should be one between 1 and 6, but got {differential_order}"
            )
        self.differential_order = differential_order

        super().__init__(cell, ns_mesh, kernel, fft_norm, ifft_norm)
        self.register_buffer(
            "_diff_coeff",
            torch.tensor(
                [  # coefficients for the difference operator
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [4 / 3, -1 / 3, 0.0, 0.0, 0.0, 0.0],
                    [3 / 2, -3 / 5, 1 / 10, 0.0, 0.0, 0.0],
                    [8 / 5, -4 / 5, 8 / 35, -1 / 35, 0.0, 0.0],
                    [5 / 3, -20 / 21, 5 / 14, -5 / 63, 1 / 126, 0.0],
                    [12 / 7, -15 / 14, 10 / 21, -1 / 7, 2 / 77, -1 / 465],
                ]
            ),
        )

    @torch.jit.export
    def update(
        self,
        cell: Optional[torch.Tensor] = None,
        ns_mesh: Optional[torch.Tensor] = None,
    ) -> None:
        # Cannot reuse code from `KSpaceFilter` by `super` because of `TorchScript`
        self._prep_kvectors(cell, ns_mesh)

        # always update the kfilter to ensure sync with potential
        self._kfilter = self._compute_influence(
            self._kvectors
        ) * self.kernel.kernel_from_k_sq(self._k_sq)

    def _compute_influence(self, kvectors: torch.Tensor) -> torch.Tensor:
        cell_dimensions = torch.linalg.norm(self.cell, dim=1)
        actual_mesh_spacing = (cell_dimensions / self.ns_mesh).reshape(1, 1, 1, 3)

        kh = kvectors * actual_mesh_spacing
        U2 = self._charge_assignment(kh)
        if self.mode == 0:
            # special (much simpler) case for point-charge potentials
            masked = torch.where(U2 == 0, 1.0, U2)
            return torch.where(U2 == 0, 0.0, torch.reciprocal(masked))

        D = self._differential_operator(kh, actual_mesh_spacing)
        D_to_4mode = torch.linalg.norm(D, dim=-1) ** (4 * self.mode)

        # Calculate (part of) the kernel See eq.30 of this paper
        # https://doi.org/10.1063/1.3000389 for your main reference, as well as the
        # paragraph below eq.31. Be careful that the reference force part is calculated
        # in the `KSpaceFilter` by calling `kernel_from_k_sq`.
        numerator = torch.sum(kvectors * D, dim=-1) ** self.mode
        denominator = U2 * D_to_4mode

        masked = torch.where(denominator == 0, 1.0, denominator)
        return torch.where(denominator == 0, 0.0, numerator / masked)

    def _differential_operator(
        self, kh: torch.Tensor, actual_mesh_spacing: torch.Tensor
    ) -> torch.Tensor:
        """
        The approximation to the differential operator ``ik``. The ``i`` is taken out
        and cancels with the prefactor ``-i`` of the reference force function (in our
        code, this is the kernel of `Potential`). See the Appendix C of this paper
        http://dx.doi.org/10.1063/1.477414.

        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, 3)
        """
        temp = torch.zeros(kh.shape, dtype=kh.dtype, device=kh.device)
        for i, coef in enumerate(
            self._diff_coeff[self.differential_order - 1][: self.differential_order]
        ):
            temp += (coef / (i + 1)) * torch.sin(kh * (i + 1))
        return temp / (actual_mesh_spacing)

    def _charge_assignment(self, kh: torch.Tensor) -> torch.Tensor:
        """
        The Fourier transformed charge assignment function divided by the volume of one
        mesh cell, in a squared form. See eq.18 and the paragraph below eq.31 of this
        paper http://dx.doi.org/10.1063/1.477414. Be aware that the volume cancels out
        with the prefactor of the assignment function (see eq.18).

        From shape (nx, ny, nz, 3) to shape (nx, ny, nz, nd)
        """
        return torch.prod(
            torch.sinc(kh / (2 * torch.pi)),
            dim=-1,
        ) ** (self.interpolation_nodes * 2)

    update.__doc__ = KSpaceFilter.update.__doc__
