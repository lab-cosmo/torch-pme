import torch


def get_ns_mesh(cell: torch.Tensor, mesh_spacing: float):
    """
    Computes the mesh size given a target mesh spacing and cell
    getting the closest powers of 2 to help with FFT.

    :param cell: torch.tensor of shape ``(3, 3)``
        Tensor specifying the real space unit cell of a structure, where ``cell[i]`` is
        the i-th basis vector
    :param mesh_spacing: float
    :param differentiable: boll

    :return: torch.tensor of length 3 containing the mesh size
    """
    basis_norms = torch.linalg.norm(cell, dim=1)
    ns_approx = basis_norms / mesh_spacing
    ns_actual_approx = 2 * ns_approx + 1  # actual number of mesh points
    # ns = [nx, ny, nz], closest power of 2 (helps for FT efficiency)
    return torch.tensor(2).pow(torch.ceil(torch.log2(ns_actual_approx)).long())


def _generate_kvectors(
    cell: torch.Tensor, ns: torch.Tensor, for_ewald: bool
) -> torch.Tensor:
    # Check that all provided parameters have the correct shapes and are consistent
    # with each other
    if cell.shape != (3, 3):
        raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")

    if ns.shape != (3,):
        raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

    if ns.device != cell.device:
        raise ValueError(
            f"`ns` and `cell` are not on the same device, got {ns.device} and "
            f"{cell.device}."
        )

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
    kxs = (bx * ns[0]) * torch.fft.fftfreq(
        ns[0], device=cell.device, dtype=cell.dtype
    ).unsqueeze(-1)
    kys = (by * ns[1]) * torch.fft.fftfreq(
        ns[1], device=cell.device, dtype=cell.dtype
    ).unsqueeze(-1)

    if for_ewald:
        kzs = (bz * ns[2]) * torch.fft.fftfreq(
            ns[2], device=cell.device, dtype=cell.dtype
        ).unsqueeze(-1)
    else:
        kzs = (bz * ns[2]) * torch.fft.rfftfreq(
            ns[2], device=cell.device, dtype=cell.dtype
        ).unsqueeze(-1)

    # then take the cartesian product (all possible combinations, same as meshgrid)
    # via broadcasting (to avoid instantiating intermediates), and sum up
    return kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]


def generate_kvectors_for_mesh(cell: torch.Tensor, ns: torch.Tensor) -> torch.Tensor:
    """
    Compute all reciprocal space vectors for Fourier space sums.

    This variant is used in combination with **mesh based calculators** using the fast
    fourier transform (FFT) algorithm.

    :param cell: torch.tensor of shape ``(3, 3)``
        Tensor specifying the real space unit cell of a structure, where ``cell[i]`` is
        the i-th basis vector
    :param ns: torch.tensor of shape ``(3,)`` and dtype int
        ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
        z-direction, respectively. For faster performance during the Fast Fourier
        Transform (FFT) it is recommended to use values of nx, ny and nz that are
        powers of 2.


    :return: torch.tensor of shape ``(nx, ny, nz, 3)`` containing all reciprocal
        space vectors that will be used in the (FFT-based) mesh calculators.
        Note that ``k_vectors[0,0,0] = [0,0,0]`` always is the zero vector.

    .. seealso::

        :func:`generate_kvectors_for_ewald` for a function to be used for Ewald
        calculators.
    """
    return _generate_kvectors(cell=cell, ns=ns, for_ewald=False)


def generate_kvectors_for_ewald(
    cell: torch.Tensor,
    ns: torch.Tensor,
) -> torch.Tensor:
    """
    Compute all reciprocal space vectors for Fourier space sums.

    This variant is used with the **Ewald calculator**, in which the sum over the
    reciprocal space vectors is performed explicitly rather than using the fast Fourier
    transform (FFT) algorithm.

    The main difference with :func:`generate_kvectors_for_mesh` is the shape of the
    output tensor (see documentation on return) and the fact that the full set of
    reciprocal space vectors is returned, rather than the FFT-optimized set that roughly
    contains only half of the vectors.

    :param cell: torch.tensor of shape ``(3, 3)``
        Tensor specifying the real space unit cell of a structure, where ``cell[i]`` is
        the i-th basis vector
    :param ns: torch.tensor of shape ``(3,)`` and dtype int
        ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
        z-direction, respectively.

    :return: torch.tensor of shape ``(n, 3)`` containing all reciprocal
        space vectors that will be used in the Ewald calculator.
        Note that ``k_vectors[0] = [0,0,0]`` always is the zero vector.

    .. seealso::

        :func:`generate_kvectors_for_mesh` for a function to be used with mesh based
        calculators like PME.
    """
    return _generate_kvectors(cell=cell, ns=ns, for_ewald=True).reshape(-1, 3)
