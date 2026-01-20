import torch
from torch.nn.utils.rnn import pad_sequence


def get_ns_mesh(cell: torch.Tensor, mesh_spacing: float):
    """
    Computes the mesh size given a target mesh spacing and cell
    getting the closest powers of 2 to help with FFT.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
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

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
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

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
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


def generate_kvectors_for_ewald_halfspace(
    cell: torch.Tensor,
    ns: torch.Tensor,
) -> torch.Tensor:
    """
    Generate half-space k-vectors exploiting Hermitian symmetry S(-k) = S*(k).

    For real-valued charge densities, the structure factor satisfies S(-k) = S*(k).
    This means computing both k and -k is redundant. This function returns only
    k-vectors in the "positive half-space". The caller should multiply the Green's
    function by 2 to compensate for the missing conjugate contributions.

    The half-space condition selects vectors where:
    - h > 0, OR
    - (h == 0 AND k > 0), OR
    - (h == 0 AND k == 0 AND l > 0)

    where (h, k, l) are the integer indices in reciprocal space. Note that k=0
    is excluded by this condition.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param ns: torch.tensor of shape ``(3,)`` and dtype int
        ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
        z-direction, respectively.

    :return: torch.tensor of shape ``(n_half, 3)`` containing the half-space
        k-vectors (excludes k=0)
    """
    # Validate inputs
    if cell.shape != (3, 3):
        raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")
    if ns.shape != (3,):
        raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

    nx = ns[0].item()
    ny = ns[1].item()
    nz = ns[2].item()

    # Compute reciprocal cell
    if cell.is_cuda:
        inverse_cell = torch.linalg.inv_ex(cell)[0]
    else:
        inverse_cell = torch.linalg.inv(cell)
    reciprocal_cell = 2 * torch.pi * inverse_cell.T
    bx, by, bz = reciprocal_cell[0], reciprocal_cell[1], reciprocal_cell[2]

    # Full frequency arrays
    fx = torch.fft.fftfreq(nx, device=cell.device, dtype=cell.dtype)
    fy = torch.fft.fftfreq(ny, device=cell.device, dtype=cell.dtype)
    fz = torch.fft.fftfreq(nz, device=cell.device, dtype=cell.dtype)

    kvecs_list = []

    # Part 1: h > 0 (signed indices 1 to nx//2, i.e., array indices 1:nx//2+1)
    # Combined with all k and all l values
    n_hpos = nx // 2
    if n_hpos >= 1:
        kxs = (bx * nx) * fx[1 : n_hpos + 1].unsqueeze(-1)  # (n_hpos, 3)
        kys = (by * ny) * fy.unsqueeze(-1)  # (ny, 3)
        kzs = (bz * nz) * fz.unsqueeze(-1)  # (nz, 3)
        kvecs1 = kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]
        kvecs_list.append(kvecs1.reshape(-1, 3))

    # Part 2: h = 0, k > 0 (signed indices 1 to ny//2 for k)
    # Combined with all l values
    n_kpos = ny // 2
    if n_kpos >= 1:
        kxs = (bx * nx) * fx[0:1].unsqueeze(-1)  # (1, 3) - h=0
        kys = (by * ny) * fy[1 : n_kpos + 1].unsqueeze(-1)  # (n_kpos, 3)
        kzs = (bz * nz) * fz.unsqueeze(-1)  # (nz, 3)
        kvecs2 = kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]
        kvecs_list.append(kvecs2.reshape(-1, 3))

    # Part 3: h = 0, k = 0, l > 0 (signed indices 1 to nz//2 for l)
    n_lpos = nz // 2
    if n_lpos >= 1:
        kxs = (bx * nx) * fx[0:1].unsqueeze(-1)  # (1, 3) - h=0
        kys = (by * ny) * fy[0:1].unsqueeze(-1)  # (1, 3) - k=0
        kzs = (bz * nz) * fz[1 : n_lpos + 1].unsqueeze(-1)  # (n_lpos, 3)
        kvecs3 = kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]
        kvecs_list.append(kvecs3.reshape(-1, 3))

    if len(kvecs_list) == 0:
        # Edge case: ns is so small that half-space is empty (e.g., ns=[1,1,1])
        return torch.zeros((0, 3), device=cell.device, dtype=cell.dtype)

    return torch.cat(kvecs_list, dim=0)


def compute_batched_kvectors(
    lr_wavelength: float,
    cells: torch.Tensor,
) -> torch.Tensor:
    r"""
    Generate half-space k-vectors for multiple systems in batches.

    Uses the half-space optimization to reduce computation by ~2x, exploiting
    Hermitian symmetry S(-k) = S*(k).

    :param lr_wavelength: Spatial resolution used for the long-range (reciprocal space)
        part of the Ewald sum. More concretely, all Fourier space vectors with a
        wavelength >= this value will be kept. If not set to a global value, it will be
        set to half the smearing parameter to ensure convergence of the
        long-range part to a relative precision of 1e-5.
    :param cells: torch.tensor of shape ``(B, 3, 3)``, where ``cells[i]`` is the i-th
        basis vector of the unit cell for system i in the batch of size B.

    :return: torch.tensor of shape ``(B, max_k, 3)`` padded k-vectors
    """
    all_kvectors = []
    k_cutoff = 2 * torch.pi / lr_wavelength
    for cell in cells:
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_float = k_cutoff * basis_norms / 2 / torch.pi
        ns = torch.ceil(ns_float).long()
        kvectors = generate_kvectors_for_ewald_halfspace(ns=ns, cell=cell)
        all_kvectors.append(kvectors)
    # Padded with zeros; G(k=0)=0 so padding doesn't contribute to the sum
    return pad_sequence(all_kvectors, batch_first=True)
