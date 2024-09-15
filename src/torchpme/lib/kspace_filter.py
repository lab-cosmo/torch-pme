from typing import Callable, List, Optional

import torch

from .kvectors import generate_kvectors_for_mesh


class KSpaceFilter:
    r"""
    Class for applying a reciprocal-space filter to a real-space mesh.
    The class combines the costruction of a reciprocal-space grid :math:`\{mathbf{k}_n\}`
    (that should be commensurate to the grid in real space, so the class takes
    the same options as :py:class:`MeshInterpolator`), the calculation of
    a scalar filter function :math:`\phi(|\mathbf{k}|^2)`, defined as a function of
    the squared norm of the reciprocal space grid points, and the application
    of the filter to a real-space function :math:`f(\mathbf{x})`,
    defined on a mesh :math:`\{mathbf{x}_n\}`.

    In practice, the application of the filter amounts to
    :math:`f\rightarrow \hat{f} \rightarrow \hat{\tilde{f}}=\hat{f} \phi \rightarrow \tilde{f}`

    See also the :ref:`example-kspace` for a demonstration of the
    functionalities of this class.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param ns_mesh: toch.tensor of shape ``(3,)``
        Number of mesh points to use along each of the three axes
    :param n_filters: int
        The dimensionality of the filter. This allows computing multiple filters that
        are applied separately to multiple real-space grids of the same shape.
    :param fft_norm: str
        The normalization applied to the forward FT. Can be
        "forward", "backward", "ortho". See :py:func:`torch:fft:rfftn`.
    :param ifft_norm: str
        The normalization applied to the inverse FT. Can be
        "forward", "backward", "ortho". See :py:func:`torch:fft:irfftn`.
    """

    def __init__(
        self,
        cell: torch.Tensor,
        ns_mesh: torch.Tensor,
        n_filters: Optional[int] = 1,
        fft_norm: Optional[str] = "ortho",
        ifft_norm: Optional[str] = "ortho",
    ):

        # TorchScript requires to initialize all attributes in __init__
        self._cell = cell
        self._ns_mesh = ns_mesh
        self._n_filters = n_filters
        self._kvectors = generate_kvectors_for_mesh(ns=ns_mesh, cell=cell)
        self._knorm_sq = torch.sum(self._kvectors**2, dim=3)
        if self._n_filters > 1:
            # creates a view with multiple copies so it is possible to apply multiple filters
            self._knorm_sq = self._knorm_sq.unsqueeze(0).expand(
                self._n_filters, -1, -1, -1
            )
        self._kfilter = torch.empty_like(self._knorm_sq)
        self._fft_norm = fft_norm
        self._ifft_norm = ifft_norm

    def set_filter_mesh(
        self, filter: List[Callable] | Callable, value_at_origin: float | None = None
    ):
        r"""
        Applies one or more scalar filter functions to the squared norms of the reciprocal
        space mesh grids, storing it to be applied to multiple real-space meshes.

        :param filter: a callable (or a list of callables) containing
            :math:`\mathbb{R}^+\rightarrow \mathbb{R} functions that will be
            applied to the tensor containing the squared norm of the k-vectors
            to prepare the convolution filter arrays
        """

        if self._n_filters > 1:
            assert hasattr(filter, "__len__")
            assert len(filter) == self._n_filters
            self._kfilter = torch.stack(
                [filter[i](self._knorm_sq[i]) for i in range(self._nfilters)]
            )
            if not value_at_origin is None:
                self._kfilter[:, 0, 0, 0] = value_at_origin
        else:
            self._kfilter = filter(self._knorm_sq)
            if not value_at_origin is None:
                self._kfilter[0, 0, 0] = value_at_origin

    def compute(
        self,
        mesh_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the k-space filter by Fourier transforming the given
        ``mesh_values`` tensor, multiplying the result by the filter array
        (that should have been previously computed with a call to
        :py:func:`set_filter_mesh`) and Fourier-transforming back
        to real space.

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

        print(mesh_values.shape, mesh_hat.shape, self._kfilter.shape)
        if (self._n_filters == 1 and mesh_hat.shape[1:] != self._kfilter.shape) or (
            self._n_filters > 1 and mesh_hat.shape != self._kfilter.shape
        ):
            raise IndexError(
                f"The shape of `mesh_values`, {mesh_values.shape} yields a FT grid"
                f"is inconsistent with the size of the k-space filter {self._kfilter.shape}"
            )

        filter_hat = mesh_hat * self._kfilter
        mesh_filter = torch.fft.irfftn(
            filter_hat,
            norm=self._ifft_norm,
            dim=dims,
            # NB: we must specify the sie of the output
            # as for certain mesh sizes the inverse FT is not
            # well-defined
            s=mesh_values.shape[1:4],
        )

        return mesh_filter
