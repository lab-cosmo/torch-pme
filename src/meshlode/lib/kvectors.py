import torch


def generate_kvectors_for_mesh(ns: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    For a given unit cell, compute all reciprocal space vectors that are used to
    perform sums in the Fourier transformed space. This variant is used in
    combination with mesh based calculators using the fast fourier transform (FFT)
    algorithm.

    :param ns: torch.tensor of shape ``(3,)`` and dtype int
        ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
        z-direction, respectively. For faster performance during the Fast Fourier
        Transform (FFT) it is recommended to use values of nx, ny and nz that are
        powers of 2.
    :param cell: torch.tensor of shape ``(3, 3)``
        Tensor specifying the real space unit cell of a structure, where cell[i] is
        the i-th basis vector

    :return: torch.tensor of shape ``(nx, ny, nz, 3)`` containing all reciprocal
        space vectors that will be used in the (FFT-based) mesh calculators.
        Note that k_vectors[0,0,0] = [0,0,0] always is the zero vector.
    """
    # Check that all provided parameters have the correct shapes and are consistent
    # with each other
    if ns.shape != (3,):
        raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

    if cell.shape != (3, 3):
        raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")

    if ns.device != cell.device:
        raise ValueError(
            f"`ns` and `cell` are not on the same device, got {ns.device} and "
            f"{cell.device}."
        )

    reciprocal_cell = 2 * torch.pi * cell.inverse().T
    bx = reciprocal_cell[0]
    by = reciprocal_cell[1]
    bz = reciprocal_cell[2]

    # Generate all reciprocal space vectors:
    # The frequencies from the fftfreq function  are of the form [0, 1/n, 2/n, ...]
    # These are then converted to [0, 1, 2, ...] by multiplying with n.
    nx = int(ns[0])
    ny = int(ns[1])
    nz = int(ns[2])

    # get the frequencies, multiply with n, then w/ the reciprocal space vectors
    kxs = (bx * ns[0]) * torch.fft.fftfreq(nx, device=ns.device).unsqueeze(-1)
    kys = (by * ns[1]) * torch.fft.fftfreq(ny, device=ns.device).unsqueeze(-1)
    # real FFT!
    kzs = (bz * ns[2]) * torch.fft.rfftfreq(nz, device=ns.device).unsqueeze(-1)
    # then take the cartesian product (all possible combinations, same as meshgrid)
    # via broadcasting (to avoid instating intermediates), and sum up
    k_vectors = kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]

    return k_vectors


def generate_kvectors_squeezed(ns: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    For a given unit cell, compute all reciprocal space vectors that are used to
    perform sums in the Fourier transformed space. This variant is used with the
    Ewald calculator, in which the sum over the reciprocal space vectors is performed
    explicitly rather than using the fast Fourier transform (FFT) algorithm.

    The main difference is the shape of the output tensor (see documentation on return)
    and the fact that the full set of reciprocal space vectors is returned, rather than
    the FFT-optimized set that roughly contains only half of the vectors.


    :param ns: torch.tensor of shape ``(3,)`` and dtype int
        ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
        z-direction, respectively.
    :param cell: torch.tensor of shape ``(3, 3)``
        Tensor specifying the real space unit cell of a structure, where cell[i] is
        the i-th basis vector

    :return: torch.tensor of shape ``(n, 3)`` containing all reciprocal
        space vectors that will be used in the Ewald calculator.
        Note that k_vectors[0] = [0,0,0] always is the zero vector.
    """
    # Check that all provided parameters have the correct shapes and are consistent
    # with each other
    if ns.shape != (3,):
        raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

    if cell.shape != (3, 3):
        raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")

    if ns.device != cell.device:
        raise ValueError(
            f"`ns` and `cell` are not on the same device, got {ns.device} and "
            f"{cell.device}."
        )

    reciprocal_cell = 2 * torch.pi * cell.inverse().T
    bx = reciprocal_cell[0]
    by = reciprocal_cell[1]
    bz = reciprocal_cell[2]

    # same as above, except for the z dimension (& the flattened output)
    nx = int(ns[0])
    ny = int(ns[1])
    nz = int(ns[2])
    kxs = (bx * ns[0]) * torch.fft.fftfreq(nx, device=ns.device).unsqueeze(-1)
    kys = (by * ns[1]) * torch.fft.fftfreq(ny, device=ns.device).unsqueeze(-1)
    kzs = (bz * ns[2]) * torch.fft.fftfreq(nz, device=ns.device).unsqueeze(-1)

    k_vectors = kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]

    return k_vectors.reshape(-1, 3)
