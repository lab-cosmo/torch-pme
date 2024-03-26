from typing import Optional

import torch


@torch.jit.script
class FourierSpaceConvolution:
    """
    Class for handling all the steps necessary to compute the convolution :math:`f*G`
    between two functions :math:`f` and :math:`G`, where the values of :math:`f` are
    provided on a discrete mesh. In practice, the convolution is performed in
    reciprocal space using the fast Fourier transform algorithm.

    Since the reciprocal space vectors used for the calculations only depend on the
    cell for a given set of hypers, the vectors are cached to reduce the computational
    cost in case multiple structures use identical cells.

    Example
    -------
    To compute the "electrostatic potential" we first have to define the cell as
    well as the grid points where we want to evaluate the potential:

    >>> import torch
    >>> L = torch.rand((1,)) * 20 + 1.0
    >>> cell = L * torch.randn((3, 3))
    >>> ns = torch.randint(1, 20, size=(4,))
    >>> n_channels, nx, ny, nz = ns
    >>> nz *= 2  # last dimension needs to be even
    >>> mesh_values = torch.randn(size=(n_channels, nx, ny, nz))

    With this definitions we just have to call the :meth:`compute` method and save the
    results

    >>> fsc = FourierSpaceConvolution()
    >>> potential = fsc.compute(mesh_values=mesh_values, cell=cell)
    """

    def __init__(self):
        self._cell_cache = torch.zeros(3, 3)
        self._ns_cache = torch.zeros(3)
        self._knorm_sq_cache = torch.empty(1)

    def generate_kvectors(self, ns: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """
        For a given unit cell, compute all reciprocal space vectors that are used to
        perform sums in the Fourier transformed space.

        :param ns: torch.tensor of shape ``(3,)``
            ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
            z-direction, respectively. For faster performance during the Fast Fourier
            Transform (FFT) it is recommended to use values of nx, ny and nz that are
            powers of 2.
        :param cell: torch.tensor of shape ``(3, 3)`` Tensor specifying the real space
            unit cell of a structure, where cell[i] is the i-th basis vector

        :return: torch.tensor of shape ``(N, 3)`` Contains all reciprocal space vectors
            that will be used during Ewald summation (or related approaches).
            ``k_vectors[i]`` contains the i-th vector, where the order has no special
            significance.
        """
        if ns.device != cell.device:
            raise ValueError(
                f"`ns` and `cell` are not on the same device, got {ns.device} and "
                f"{cell.device}."
            )

        if ns.shape != (3,):
            raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

        if cell.shape != (3, 3):
            raise ValueError(
                f"cell of shape {list(cell.shape)} should be of shape (3, 3)"
            )

        # Define basis vectors of the reciprocal cell
        reciprocal_cell = 2 * torch.pi * cell.inverse().T
        bx = reciprocal_cell[0]
        by = reciprocal_cell[1]
        bz = reciprocal_cell[2]

        # Generate all reciprocal space vectors
        nxs_1d = ns[0] * torch.fft.fftfreq(ns[0], device=ns.device)
        nys_1d = ns[1] * torch.fft.fftfreq(ns[1], device=ns.device)
        nzs_1d = ns[2] * torch.fft.rfftfreq(ns[2], device=ns.device)  # real FFT
        nxs, nys, nzs = torch.meshgrid(nxs_1d, nys_1d, nzs_1d, indexing="ij")
        nxs = nxs.reshape((int(ns[0]), int(ns[1]), len(nzs_1d), 1))
        nys = nys.reshape((int(ns[0]), int(ns[1]), len(nzs_1d), 1))
        nzs = nzs.reshape((int(ns[0]), int(ns[1]), len(nzs_1d), 1))
        k_vectors = nxs * bx + nys * by + nzs * bz

        return k_vectors

    def kernel_func(
        self,
        ksq: torch.Tensor,
        potential_exponent: int = 1,
        atomic_smearing: float = 0.2,
    ) -> torch.Tensor:
        """
        Fourier transform of the Coulomb potential or more general effective
        :math:`1/r^p` potentials with additional ``atomic_smearing`` to remove the
        singularity at the origin.

        :param ksq: torch.tensor of shape ``(N,)`` Squared norm of the k-vectors
        :param potential_exponent: Exponent of the effective :math:`1/r^p` decay
        :param atomic_smearing: Broadening of the :math:`1/r^p` decay close to the
            origin

        :return: torch.tensor of shape ``(N,)`` with the values of the kernel function
            G(k) evaluated at the provided (squared norms of the) k-vectors
        """
        if potential_exponent == 1:
            return 4 * torch.pi / ksq * torch.exp(-0.5 * atomic_smearing**2 * ksq)
        elif potential_exponent == 0:
            return torch.exp(-0.5 * atomic_smearing**2 * ksq)
        else:
            raise ValueError("Only potential exponents 0 and 1 are supported!")

    def value_at_origin(
        self, potential_exponent: int = 1, atomic_smearing: Optional[float] = 0.2
    ) -> float:
        """
        Since the kernel function in reciprocal space typically has a (removable)
        singularity at k=0, the value at that point needs to be specified explicitly.

        :param potential_exponent: Exponent of the effective :math:`1/r^p` decay
        :param atomic_smearing: Broadening of the :math:`1/r^p` decay close to the
            origin

        :return: float of G(k=0), the value of the kernel function at the origin.
        """
        if potential_exponent == 1:
            return 0.0
        elif potential_exponent == 0:
            return 1.0
        else:
            raise ValueError("Only potential exponents 0 and 1 are supported")

    def compute(
        self,
        mesh_values: torch.Tensor,
        cell: torch.Tensor,
        potential_exponent: int = 1,
        atomic_smearing: float = 0.2,
    ) -> torch.Tensor:
        """
        Compute the "electrostatic potential" from the density defined
        on a discrete mesh.

        :param mesh_values: torch.tensor of shape ``(n_channels, nx, ny, nz)``
            The values of the density defined on a mesh.
        :param cell: torch.tensor of shape ``(3, 3)`` Tensor specifying the real space
            unit cell of a structure, where cell[i] is the i-th basis vector
        :param potential_exponent: int
            The exponent in the :math:`1/r^p` decay of the effective potential,
            where :math:`p=1` corresponds to the Coulomb potential,
            and :math:`p=0` is set as Gaussian atomic_smearing.
        :param atomic_smearing: float
            Width of the Gaussian atomic_smearing (for the Coulomb potential).

        :returns: torch.tensor of shape ``(n_channels, nx, ny, nz)``
            The potential evaluated on the same mesh points as the provided
            density.
        """
        if mesh_values.dim() != 4:
            raise ValueError(
                "`mesh_values` needs to be a 4 dimensional tensor, got "
                f"{mesh_values.dim()}"
            )

        if cell.shape != (3, 3):
            raise ValueError(
                f"cell of shape {list(cell.shape)} should be of shape (3, 3)"
            )

        if mesh_values.device != cell.device:
            raise ValueError(
                "`mesh_values` and `cell` are on different devices, got "
                f"{mesh_values.device} and {cell.device}"
            )

        dtype = cell.dtype
        device = cell.device

        # Get shape information from mesh
        _, nx, ny, nz = mesh_values.shape
        ns = torch.tensor([nx, ny, nz], device=mesh_values.device)

        # Use chached values if cell and number of mesh points have not changed since
        # last call.
        self._cell_cache = self._cell_cache.to(dtype=dtype, device=device)
        self._ns_cache = self._ns_cache.to(dtype=dtype, device=device)

        same_cell = torch.allclose(cell, self._cell_cache, atol=1e-15, rtol=1e-15)
        if torch.all(ns == self._ns_cache) and same_cell:
            knorm_sq = self._knorm_sq_cache.to(dtype=dtype, device=device)
        else:
            # Get the relevant reciprocal space vectors (k-vectors)
            # and compute their norm.
            kvectors = self.generate_kvectors(ns=ns, cell=cell)
            knorm_sq = torch.sum(kvectors**2, dim=3)

            # Store values for the cache. We do not clone the arrays because we only
            # read the value and do not perform any inplace operations
            self._cell_cache = cell
            self._ns_cache = ns
            self._knorm_sq_cache = knorm_sq

        # G(k) is the Fourier transform of the Coulomb potential
        # generated by a Gaussian charge density
        # We remove the singularity at k=0 by explicitly setting its
        # value to be equal to zero. This mathematically corresponds
        # to the requirement that the net charge of the cell is zero.
        # G = kernel_func(knorm_sq)
        G = self.kernel_func(
            knorm_sq,
            potential_exponent=potential_exponent,
            atomic_smearing=atomic_smearing,
        )
        G[0, 0, 0] = self.value_at_origin(
            potential_exponent=potential_exponent, atomic_smearing=atomic_smearing
        )

        # Fourier transforms consisting of the following substeps:
        # 1. Fourier transform the density
        # 2. multiply by kernel in k-space
        # 3. transform back
        # For the Fourier transforms, we use the normalization conditions
        # that do not introduce any extra factors of 1/n_mesh.
        # This is why the forward transform (fft) is called with the
        # normalization option 'backward' (the convention in which 1/n_mesh
        # is in the backward transformation) and vice versa for the
        # inverse transform (irfft).
        volume = cell.det()
        dims = (1, 2, 3)  # dimensions along which to Fourier transform
        mesh_hat = torch.fft.rfftn(mesh_values, norm="backward", dim=dims)
        potential_hat = mesh_hat * G
        potential_mesh = (
            torch.fft.irfftn(potential_hat, norm="forward", dim=dims) / volume
        )

        return potential_mesh
