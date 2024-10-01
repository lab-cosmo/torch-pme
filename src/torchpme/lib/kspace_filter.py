import torch

from .kvectors import generate_kvectors_for_mesh


class KSpaceKernel(torch.nn.Module):
    r"""
    Base class defining the interface for a reciprocal-space kernel helper.

    Provides an interface to compute the reciprocal-space convolution kernel.
    Parameters of the kernel in derived classes should be defined and stored in the
    ``__init__`` method.
    The :py:class:`RangeSeparatedPotential` classes inherits from this class,
    as the Fourier-space kernel functionality is used to compute potentials
    using Fourier transforms.

    NB: we need this slightly convoluted way of implementing what often amounts
    to a simple, pure function of :math:`|\mathbf{k}|^2` in order to be able to
    provide a customizable filter class that can be jitted/compiled.
    """

    def __init__(self):
        super().__init__()

    def from_k(self, k: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the reciprocal-space kernel on a grid of k points given a
        tensor containing :math:`|\mathbf{k}|`.

        :param k: torch.tensor containing the k vector moduli at which the kernel
            is to be evaluated.
        """
        return self.from_k_sq(k**2)

    def from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the reciprocal-space kernel on a grid of k points given a
        tensor containing :math:`|\mathbf{k}|^2`.

        :param k_sq: torch.tensor containing the squared k vector moduli
            at which the kernel is to be evaluated.
        """
        raise NotImplementedError(
            f"from_k_sq is not implemented for '{self.__class__.__name__}'"
        )


class KSpaceFilter(torch.nn.Module):
    r"""
    Apply a reciprocal-space filter to a real-space mesh.

    The class combines the costruction of a reciprocal-space grid
    :math:`\{mathbf{k}_n\}`
    (that should be commensurate to the grid in real space, so the class takes
    the same options as :py:class:`MeshInterpolator`), the calculation of
    a scalar filter function :math:`\phi(|\mathbf{k}|^2)`, defined as a function of
    the squared norm of the reciprocal space grid points, and the application
    of the filter to a real-space function :math:`f(\mathbf{x})`,
    defined on a mesh :math:`\{mathbf{x}_n\}`.

    In practice, the application of the filter amounts to
    :math:`f\rightarrow \hat{f} \rightarrow \hat{\tilde{f}}=
    \hat{f} \phi \rightarrow \tilde{f}`

    See also the :ref:`example-kspace` for a demonstration of the
    functionalities of this class.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param ns_mesh: toch.tensor of shape ``(3,)``
        Number of mesh points to use along each of the three axes
    :param kernel: KSpaceKernel
        A :py:class:`KSpaceKernel`-derived class providing a
        ``from_k_sq`` method that
        evaluates :math:`\psi` given the square modulus of
        the k-space mesh points. Note that
        :py:class:`RangeSeparatedPotential` classes inherit from
        :py:class:`KSpaceKernel` and so can also be used as a kernel.

    :param fft_norm: str
        The normalization applied to the forward FT. Can be
        "forward", "backward", "ortho". See :py:func:`torch:fft:rfftn`
    :param ifft_norm: str
        The normalization applied to the inverse FT. Can be
        "forward", "backward", "ortho". See :py:func:`torch:fft:irfftn`
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

        self._kernel = kernel
        self.update_mesh(cell, ns_mesh)

    @torch.jit.export
    def update_filter(self):
        r"""
        Applies one or more scalar filter functions to the squared norms of the
        reciprocal space mesh grids, storing it so it can be applied multiple times.
        Uses the :py:func:`KSpaceKernel.from_k_sq` method of the
        :py:class:`KSpaceKernel`-derived object provided upon initialization
        to compute the kernel values over the grid points.
        """
        self._kfilter = self._kernel.from_k_sq(self._knorm_sq)

    @torch.jit.export
    def update_mesh(self, cell: torch.Tensor, ns_mesh: torch.Tensor):
        """
        Update the k-space mesh vectors.

        Should have a size consistent with that of the mesh used to
        store the real-space functions that will be filtered.

        :param cell: torch.tensor of shape ``(3, 3)``, where `
            `cell[i]`` is the i-th basis vector of the unit cell
        :param ns_mesh: toch.tensor of shape ``(3,)``
            Number of mesh points to use along each of the three axes
        """
        # Check that the provided parameters match the specifications
        if cell.shape != (3, 3):
            raise ValueError(
                f"cell of shape {list(cell.shape)} should be of shape (3, 3)"
            )
        if ns_mesh.shape != (3,):
            raise ValueError(f"shape {list(ns_mesh.shape)} of `ns_mesh` has to be (3,)")
        if cell.device != ns_mesh.device:
            raise ValueError(
                "`cell` and `ns_mesh` are on different devices, got "
                f"{cell.device} and {ns_mesh.device}"
            )

        self._cell = cell
        self._ns_mesh = ns_mesh
        self._kvectors = generate_kvectors_for_mesh(ns=ns_mesh, cell=cell)
        self._knorm_sq = torch.sum(self._kvectors**2, dim=3)
        # also updates filter to reduce the risk it'd go out of sync
        self.update_filter()

    @torch.jit.export
    def compute(
        self,
        mesh_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the k-space filter by Fourier transforming the given
        ``mesh_values`` tensor, multiplying the result by the filter array
        (that should have been previously computed with a call to
        :py:func:`update_filter`) and Fourier-transforming back
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

    def forward(self, cell: torch.Tensor, mesh: torch.Tensor):
        """
        Performs a full k-space convolution step.

        The default forward call for `KSpaceFilter` combines
        the construction or update of the mesh (including the
        calculation of the filter values) and the Fourier
        convolution with a given grid.

        The size of the mesh is inferred from the input mesh
        size.
        """
        self.update_mesh(cell, torch.tensor(mesh.shape[-3:]))

        return self.compute(mesh)
