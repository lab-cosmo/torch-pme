import torch


class Kvectors:
    """Class for handling computation of the reciprocal space vectors (kvectors).

    Reciprocal space vectors only depend on the ``cell`` and the mesh and therefore the
    vectors are cached to reduce the computational cost in case multiple structures use
    identical cells.

    :param for_ewald: With this option the :meth:`compute` methods will return the
        output tensors in the correct format for the Ewald calculator. For the default
        value of ``for_ewald=False`` the shape and values are suitable in combination
        with mesh based calculators using the fast fourier transform (FFT) algorithm.

    Example
    -------
    To compute the vectors for an PME calculator we first have to define the cell as
    well as the grid points:

    >>> import torch
    >>> ns = torch.randint(1, 20, size=(3,))
    >>> L = torch.rand((1,)) * 20 + 1.0
    >>> cell = L * torch.randn((3, 3))

    With this definitions we just have to call the :meth:`compute` method and
    save the results

    >>> kvector_generator = Kvectors()
    >>> kvectors = kvector_generator.compute(ns=ns, cell=cell)
    """

    def __init__(self, for_ewald: bool = False):

        self.for_ewald = for_ewald

        # TorchScript requires to initialize all attributes in __init__
        self._cell_cache = torch.zeros(3, 3)
        self._ns_cache = torch.zeros(3)
        self._kvectors_cache = torch.zeros(1)

    def _generate_kvectors(self, ns: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        if cell.is_cuda:
            # use function that does not synchronize with the CPU
            inverse_cell = torch.linalg.inv_ex(cell)[0]
        else:
            inverse_cell = torch.linalg.inv(cell)

        reciprocal_cell = 2 * torch.pi * inverse_cell.T
        bx = reciprocal_cell[0]
        by = reciprocal_cell[1]
        bz = reciprocal_cell[2]

        # Generate all reciprocal space vectors from real FFT!
        # The frequencies from the fftfreq function  are of the form [0, 1/n, 2/n, ...]
        # These are then converted to [0, 1, 2, ...] by multiplying with n.
        # get the frequencies, multiply with n, then w/ the reciprocal space vectors
        kxs = bx * ns[0] * torch.fft.fftfreq(ns[0], device=ns.device).unsqueeze(-1)
        kys = by * ns[1] * torch.fft.fftfreq(ns[1], device=ns.device).unsqueeze(-1)

        if self.for_ewald:
            kzs = bz * ns[2] * torch.fft.fftfreq(ns[2], device=ns.device).unsqueeze(-1)
        else:
            kzs = bz * ns[2] * torch.fft.rfftfreq(ns[2], device=ns.device).unsqueeze(-1)

        # then take the cartesian product (all possible combinations, same as meshgrid)
        # via broadcasting (to avoid instantiating intermediates), and sum up
        return kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]

    def _get_kvectors(self, ns: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        # Check that all provided parameters have the correct shapes and are consistent
        # with each other
        if ns.shape != (3,):
            raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

        if cell.shape != (3, 3):
            raise ValueError(
                f"cell of shape {list(cell.shape)} should be of shape (3, 3)"
            )

        if ns.device != cell.device:
            raise ValueError(
                f"`ns` and `cell` are not on the same device, got {ns.device} and "
                f"{cell.device}."
            )

        dtype = cell.dtype
        device = cell.device

        # Use chached values if cell and number of mesh points have not changed since
        # last call.
        self._cell_cache = self._cell_cache.to(dtype=dtype, device=device)
        self._ns_cache = self._ns_cache.to(dtype=dtype, device=device)

        same_cell = torch.allclose(cell, self._cell_cache, atol=1e-15, rtol=1e-15)
        if torch.all(ns == self._ns_cache) and same_cell:
            kvectors = self._kvectors_cache
        else:
            kvectors = self._generate_kvectors(ns=ns, cell=cell)

            # Store values for the cache. We do not clone the arrays because we only
            # read the value and do not perform any inplace operations
            self._cell_cache = cell
            self._ns_cache = ns
            self._kvectors_cache = kvectors

        return kvectors

    def compute(self, ns: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """
        For a given unit cell, compute all reciprocal space vectors that are used to
        perform sums in the Fourier transformed space.

        :param ns: torch.tensor of shape ``(3,)`` and dtype int ``ns = [nx, ny, nz]``
            contains the number of mesh points in the x-, y- and z-direction,
            respectively. For faster performance during the Fast Fourier Transform (FFT)
            it is recommended to use values of nx, ny and nz that are powers of 2.
        :param cell: torch.tensor of shape ``(3, 3)`` Tensor specifying the real space
            unit cell of a structure, where cell[i] is the i-th basis vector

        :return: torch.tensor of shape ``(nx, ny, nz, 3)`` containing all reciprocal
            space vectors that will be used in the (FFT-based) mesh calculators. Note
            that k_vectors[0,0,0] = [0,0,0] always is the zero vector.
        """
        kvectors = self._get_kvectors(ns=ns, cell=cell)

        if self.for_ewald:
            return kvectors.reshape(-1, 3)
        else:
            return kvectors
