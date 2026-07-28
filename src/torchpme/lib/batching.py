r"""
Host-side helpers to assemble a *tiled* batch for
:meth:`EwaldCalculator.forward_batched <torchpme.EwaldCalculator.forward_batched>`.

The batched Ewald evaluation follows two ideas (see the ``forward_batched``
documentation for the evaluation itself):

1. A single target half-space k-vector count ``num_k`` is the knob that balances the
   real- and reciprocal-space work for every system in a batch. It fixes, per system
   :math:`b`, the reciprocal resolution
   :math:`\lambda_b = (\prod_i L_{b,i} / (2\,\mathrm{num\_k}))^{1/3}`, the smearing
   :math:`\sigma_b = c_\sigma \lambda_b` and the real-space cutoff
   :math:`r_c(b) = c_r \sigma_b` (use :func:`ewald_params_from_num_k` to compute the
   cutoff at which each system's neighbor list must be built; the factors
   :math:`c_\sigma` and :math:`c_r` are adjustable and default to 2 and 5).
2. The reciprocal-space atom :math:`\times` k-vector sum is block-diagonal across
   systems. :func:`prepare_tiled_batch` lays atoms out in a flat sum-padded array,
   pads every system's k-vectors to a common ``K_pad`` and enumerates only the
   on-diagonal ``(system, atom_tile, k_tile)`` work blocks in a dispatch table, so the
   dense ``[n_systems, max_atoms, max_kvectors]`` rectangle is never materialized.

All functions here run on the host once per batch (or once per dataset for
:func:`ewald_params_from_num_k`); nothing needs to be differentiable. The collated
tensors are returned on the device of the inputs.

The tiling *shapes* — the per-axis k-grid extents, the padded k-count ``K_pad``, the
length of the flat atom layout and of the dispatch table — are Python integers, so
:func:`prepare_tiled_batch` has to read a few values back from the device. It copies
the per-system metadata in one go, so a batch of 3D-periodic and non-periodic systems
synchronizes exactly once however many systems it holds; 2D slabs cost a little more,
since shrinking their cell reads a few scalars back per slab. Collating on CPU tensors
— in a dataloader worker, say — and moving the batch to the device afterwards avoids
the transfers altogether.
"""

import warnings

import torch

from .._utils import _validate_parameters

__all__ = [
    "ewald_params_from_num_k",
    "prepare_tiled_batch",
    "shrink_2d_cell",
]


def shrink_2d_cell(
    cell: torch.Tensor, periodic: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    r"""
    Replace the vacuum-padded axis of a 2D-periodic cell by a shorter, physics-safe one.

    For a slab with a large vacuum gap, the length of the non-periodic cell vector
    inflates the cell-length product and with it the derived smearing, k-vector count
    and real-space cutoff. This helper shrinks the non-periodic vector to
    ``thickness + 1.5 * L_max`` (with ``L_max`` the longest periodic vector), which
    keeps the residual error of the slab treatment at :math:`\approx e^{-3\pi}`.
    The cell is returned unchanged if it is not 2D-periodic or already short enough.

    A zero non-periodic cell vector (the ``metatomic`` convention for non-periodic
    directions) is supported: the vacuum axis is then synthesized along the normal of
    the periodic plane at the shrunk height.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param periodic: torch.tensor of shape ``(3,)`` and dtype bool
    :param positions: torch.tensor of shape ``(n_atoms, 3)``
    :return: torch.tensor of shape ``(3, 3)`` with the (possibly) shrunk cell

    Example
    -------
    A slab of thickness 1.5 in a cell with 100 units of vacuum keeps its periodic
    vectors, while the vacuum axis shrinks to ``1.5 + 1.5 * 4 = 7.5``:

    >>> import torch
    >>> cell = torch.diag(torch.tensor([4.0, 4.0, 100.0]))
    >>> positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 1.5]])
    >>> periodic = torch.tensor([True, True, False])
    >>> shrink_2d_cell(cell, periodic, positions)
    tensor([[4.0000, 0.0000, 0.0000],
            [0.0000, 4.0000, 0.0000],
            [0.0000, 0.0000, 7.5000]])

    """
    if int(periodic.sum()) != 2:
        return cell

    axis = int(torch.argmax((~periodic).to(torch.int64)))
    vac = cell[axis]
    height = torch.linalg.norm(vac)
    if height > 0:
        normal = vac / height
    else:
        # zero vacuum row: build the axis from the periodic-plane normal
        r1 = cell[(axis + 1) % 3]
        r2 = cell[(axis + 2) % 3]
        normal = torch.linalg.cross(r1, r2)
        normal = normal / torch.linalg.norm(normal)

    # slab thickness as the extent of the atoms along the vacuum-axis direction
    z = positions @ normal
    thickness = z.max() - z.min()

    lengths = torch.linalg.norm(cell, dim=-1)
    # longest periodic vector (a generator expression here would not be scriptable)
    l_max = torch.cat([lengths[:axis], lengths[axis + 1 :]]).max()
    h_min = thickness + 1.5 * l_max

    if height > 0 and height <= h_min:
        return cell

    new_cell = cell.clone()
    new_cell[axis] = normal * h_min
    return new_cell


def ewald_params_from_num_k(
    cell: torch.Tensor,
    periodic: torch.Tensor,
    num_k: int,
    smearing_factor: float = 2.0,
    cutoff_factor: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Derive the per-system Ewald parameters from the target k-vector count ``num_k``.

    Inverting the half-space k-count formula
    :math:`\mathrm{num\_k} \approx \tfrac{1}{2}\prod_i \lceil L_i/\lambda\rceil` gives
    the reciprocal resolution :math:`\lambda`, from which the smearing
    :math:`\sigma = c_\sigma \lambda` and the real-space cutoff
    :math:`r_c = c_r \sigma` follow. ``r_c`` is the radius at which the system's
    neighbor list should be built before calling :func:`prepare_tiled_batch`; since it
    grows with the cell size, it can also be evaluated once for the largest cell of a
    dataset and used for every structure.

    The two factors :math:`c_\sigma = \mathrm{smearing\_factor}` and
    :math:`c_r = \mathrm{cutoff\_factor}` control the two truncation errors of the
    split, and their defaults (2 and 5) are deliberately conservative rules of thumb:
    :math:`c_\sigma` sets how well the k-grid of resolution :math:`\lambda` resolves a
    Gaussian of width :math:`\sigma` (the reciprocal-space error decreases with
    :math:`c_\sigma`, at the price of a larger cutoff), while :math:`c_r` sets the
    real-space error :math:`\propto \mathrm{erfc}(c_r/\sqrt{2})` (increasing it makes
    the neighbor list, and hence the real-space work, more expensive). The default
    :math:`c_r = 5` matches the rule of thumb documented for
    :class:`torchpme.EwaldCalculator`, which suggests a smearing of one fifth of the
    neighbor-list cutoff. To pick parameters from a target accuracy for a *single*
    structure instead, use :func:`torchpme.tuning.tune_ewald`.

    .. warning::

        Pass the same ``smearing_factor`` here and to :func:`prepare_tiled_batch`.
        The value used here defines the cutoff your neighbor lists are built at, while
        the one given to :func:`prepare_tiled_batch` defines the smearing the
        evaluation splits at; if the two disagree, the real- and reciprocal-space parts
        no longer match and the result is silently inaccurate. This shared factor is
        deliberately the only thing tying the two calls together:
        :func:`prepare_tiled_batch` derives :math:`\lambda` and :math:`\sigma` again
        from its own (2D-shrunk) cells, on host data it has to read anyway, so handing
        the values computed here back to it would save nothing.

    For 2D-periodic systems pass the *effective* (vacuum-shrunk) cell, see
    :func:`shrink_2d_cell`. Non-periodic systems (``periodic`` all ``False``) have no
    reciprocal part; they get placeholder values ``lambda = 0``, ``sigma = 1``,
    ``r_c = 0`` (their real-space part is the bare potential over the caller-provided
    pair list).

    :param cell: torch.tensor of shape ``(3, 3)`` or batched ``(n_systems, 3, 3)``
    :param periodic: torch.tensor of dtype bool and shape ``(3,)`` or batched
        ``(n_systems, 3)``
    :param num_k: target half-space k-vector count per system
    :param smearing_factor: ratio :math:`\sigma/\lambda` of the smearing to the
        reciprocal resolution
    :param cutoff_factor: ratio :math:`r_c/\sigma` of the real-space cutoff to the
        smearing
    :return: tuple ``(lambda, sigma, cutoff)``, each of shape ``()`` for a single
        system or ``(n_systems,)`` for a batch

    Example
    -------
    >>> import torch
    >>> cell = 4.0 * torch.eye(3)
    >>> periodic = torch.tensor([True, True, True])
    >>> lam, sigma, cutoff = ewald_params_from_num_k(cell, periodic, num_k=200)
    >>> print(f"{float(lam):.3f} {float(sigma):.3f} {float(cutoff):.3f}")
    0.543 1.086 5.429

    A larger ``cutoff_factor`` trades real-space work for accuracy, without changing
    the smearing or the number of k-vectors:

    >>> _, sigma, cutoff = ewald_params_from_num_k(
    ...     cell, periodic, num_k=200, cutoff_factor=8.0
    ... )
    >>> print(f"{float(sigma):.3f} {float(cutoff):.3f}")
    1.086 8.686

    """
    if num_k < 1:
        raise ValueError(f"`num_k` must be a positive integer, got {num_k}")
    if smearing_factor <= 0:
        raise ValueError(f"`smearing_factor` must be positive, got {smearing_factor}")
    if cutoff_factor <= 0:
        raise ValueError(f"`cutoff_factor` must be positive, got {cutoff_factor}")

    single = cell.dim() == 2
    cells = cell.unsqueeze(0) if single else cell
    periodics = periodic.unsqueeze(0) if single else periodic

    lengths = torch.linalg.norm(cells, dim=-1)
    lam = (lengths.prod(dim=-1) / (2.0 * num_k)) ** (1.0 / 3.0)
    sigma = smearing_factor * lam
    cutoff = cutoff_factor * sigma

    nonperiodic = ~periodics.any(dim=-1)
    lam = torch.where(nonperiodic, torch.zeros_like(lam), lam)
    sigma = torch.where(nonperiodic, torch.ones_like(sigma), sigma)
    cutoff = torch.where(nonperiodic, torch.zeros_like(cutoff), cutoff)

    if single:
        return lam[0], sigma[0], cutoff[0]
    return lam, sigma, cutoff


def _integer_kgrid(ns: list[int], halfspace: bool) -> torch.Tensor:
    """
    Integer k-grid (FFT-frequency convention) for per-axis extents ``ns``.

    With ``halfspace=True`` only one of each ±k pair is kept, following the
    lexicographic rule ``(kx>0) | (kx==0 & ky>0) | (kx==0 & ky==0 & kz>0)``, which
    also excludes k = 0; each kept vector then stands in for its -k partner
    (degeneracy factor 2 in the evaluation). With ``halfspace=False`` the full grid
    (including k = 0, which the potentials map to a zero kernel) is returned.
    """
    freqs = [(torch.fft.fftfreq(n) * n).round().to(torch.long) for n in ns]
    kx, ky, kz = torch.meshgrid(freqs[0], freqs[1], freqs[2], indexing="ij")
    grid = torch.stack([kx.flatten(), ky.flatten(), kz.flatten()], dim=-1)

    if not halfspace:
        return grid

    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]
    keep = (x > 0) | ((x == 0) & (y > 0)) | ((x == 0) & (y == 0) & (z > 0))
    return grid[keep]


def prepare_tiled_batch(
    positions: list[torch.Tensor],
    charges: list[torch.Tensor],
    cells: list[torch.Tensor],
    periodic: list[torch.Tensor],
    neighbor_indices: list[torch.Tensor],
    neighbor_distances: list[torch.Tensor],
    num_k: int,
    block_atoms: int = 32,
    block_kvecs: int = 128,
    halfspace: bool = True,
    k_pad_fraction: float = 0.1,
    smearing_factor: float = 2.0,
    smearing: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    r"""
    Collate a list of systems into a concatenated batch plus the static tiling data
    consumed by :meth:`EwaldCalculator.forward_batched
    <torchpme.EwaldCalculator.forward_batched>`.

    Each system may be 3D-periodic, 2D-periodic (a slab with vacuum along the
    non-periodic axis; the effective cell is shrunk internally, see
    :func:`shrink_2d_cell`) or non-periodic. 1D periodicity is not supported. The
    neighbor list of every *periodic* system must have been built beforehand at that
    system's own cutoff ``r_c(b)`` from :func:`ewald_params_from_num_k` (or any larger
    radius); non-periodic systems provide the pair list over which the *bare*
    (non-range-separated) potential is summed, typically all pairs.

    All systems must share the dtype and the device of their tensors; the returned
    tensors live on that same device, ready to be passed to the calculator. The index
    arithmetic itself runs on the host: since it decides the shapes of the tiling, it
    needs the cells and periodicities as host integers, and copies them from the device
    once per batch rather than once per system. Only 2D slabs add to that, through the
    scalars :func:`shrink_2d_cell` reads back. Collating on CPU tensors and moving the
    batch to the device afterwards avoids the transfers entirely.

    Nothing in the returned ``tiling`` carries a gradient: it is static data derived
    from the cells, and the smearing it holds is a convergence parameter rather than a
    function of the geometry. The ``batch`` entries are the caller's own tensors, so
    gradients flow through them as usual.

    :param positions: per-system tensors of shape ``(n_atoms_b, 3)``
    :param charges: per-system tensors of shape ``(n_atoms_b, n_channels)``
    :param cells: per-system tensors of shape ``(3, 3)``
    :param periodic: per-system bool tensors of shape ``(3,)``
    :param neighbor_indices: per-system tensors of shape ``(n_pairs_b, 2)`` with
        *local* atom indices (offset internally)
    :param neighbor_distances: per-system tensors of shape ``(n_pairs_b,)``
    :param num_k: target half-space k-vector count per system (the accuracy/cost knob)
    :param block_atoms: atom-tile size ``BM`` of the reciprocal-space kernel
    :param block_kvecs: k-tile size ``BK`` of the reciprocal-space kernel
    :param halfspace: keep only one of each ±k pair (with degeneracy factor 2)
    :param k_pad_fraction: padding window of the common k-count: the k-axis is padded
        to at least ``(1 + k_pad_fraction) * num_k`` vectors so that batches sharing
        the same ``num_k`` get identical shapes
    :param smearing_factor: ratio of the smearing to the reciprocal resolution, see
        :func:`ewald_params_from_num_k`; must match the value used to compute the
        neighbor-list cutoffs
    :param smearing: optional override of the derived per-system smearing (a 0-dim
        tensor or one of shape ``(n_systems,)``), which ignores ``smearing_factor``;
        the k-grids are still sized by ``num_k``. It is consumed on the host, so pass
        it on the CPU to avoid one more device-to-host copy.
    :return: tuple ``(batch, tiling)`` of two dictionaries of tensors. ``batch`` holds
        the concatenated inputs of ``forward_batched`` (``positions``, ``charges``,
        ``cell``, ``periodic``, ``system_index``, ``neighbor_indices``,
        ``neighbor_distances``); ``tiling`` holds the static index tensors and
        parameters (its integer metadata is stored as 0-dim tensors so that it stays a
        ``Dict[str, Tensor]`` for TorchScript).

    Example
    -------
    Two 3D-periodic systems of different size, collated and evaluated in one call. In
    practice the per-system pair lists come from a neighbor-list code (such as
    ``vesin``) run at the cutoff of :func:`ewald_params_from_num_k`; here they are
    written out explicitly:

    >>> import torch
    >>> import torchpme
    >>> positions = [
    ...     torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    ...     torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    ... ]
    >>> charges = [torch.tensor([[1.0], [-1.0]])] * 2
    >>> cells = [torch.eye(3), 2.0 * torch.eye(3)]
    >>> periodic = [torch.tensor([True, True, True])] * 2
    >>> neighbor_indices = [torch.tensor([[0, 1]])] * 2
    >>> neighbor_distances = [torch.tensor([0.8660]), torch.tensor([1.7321])]
    >>> batch, tiling = prepare_tiled_batch(
    ...     positions=positions,
    ...     charges=charges,
    ...     cells=cells,
    ...     periodic=periodic,
    ...     neighbor_indices=neighbor_indices,
    ...     neighbor_distances=neighbor_distances,
    ...     num_k=200,
    ... )

    The smearing is derived per system from ``num_k``, so the larger cell gets the
    larger smearing:

    >>> print(tiling["sigma"])
    tensor([0.2714, 0.5429])

    ``forward_batched`` returns the concatenated per-atom potentials; split them per
    system with ``batch["system_index"]``:

    >>> calculator = torchpme.EwaldCalculator(
    ...     torchpme.CoulombPotential(smearing=1.0), lr_wavelength=1.0
    ... )
    >>> potentials = calculator.forward_batched(
    ...     batch["charges"],
    ...     batch["cell"],
    ...     batch["positions"],
    ...     batch["neighbor_indices"],
    ...     batch["neighbor_distances"],
    ...     batch["system_index"],
    ...     batch["periodic"],
    ...     tiling,
    ... )
    >>> print(potentials.shape, batch["system_index"])
    torch.Size([4, 1]) tensor([0, 0, 1, 1])

    """
    n_systems = len(positions)
    if n_systems == 0:
        raise ValueError("Cannot collate an empty list of systems")
    if not (
        len(charges)
        == len(cells)
        == len(periodic)
        == len(neighbor_indices)
        == len(neighbor_distances)
        == n_systems
    ):
        raise ValueError("All per-system input lists must have the same length")
    if block_atoms < 1 or block_kvecs < 1:
        raise ValueError("`block_atoms` and `block_kvecs` must be positive")

    # the shapes, dtypes and devices of a single system are exactly what the
    # calculators check on every forward call, so reuse the same helper here
    for b in range(n_systems):
        _validate_parameters(
            charges=charges[b],
            cell=cells[b],
            positions=positions[b],
            neighbor_indices=neighbor_indices[b],
            neighbor_distances=neighbor_distances[b],
            periodic=periodic[b],
        )

    # what remains are the invariants across systems
    dtype = positions[0].dtype
    device = positions[0].device
    n_channels = charges[0].shape[1]
    for b in range(1, n_systems):
        if positions[b].dtype != dtype or positions[b].device != device:
            raise ValueError(
                "All systems must share the dtype and device of the first system, but "
                f"system {b} has {positions[b].dtype} on {positions[b].device} instead "
                f"of {dtype} on {device}"
            )
        if charges[b].shape[1] != n_channels:
            raise ValueError(
                "All systems must have the same number of charge channels, but system "
                f"{b} has {charges[b].shape[1]} instead of {n_channels}"
            )

    # --- host-side metadata ---
    # everything that decides a *shape* below (the periodicity of a system, its k-grid
    # extents, the length of the flat atom layout) has to be a host integer, so the
    # per-system metadata is copied to the CPU in one go rather than read back value by
    # value; the collated tensors themselves never leave the device
    cells_batch = torch.stack(list(cells))
    periodic_batch = torch.stack(list(periodic))
    packed = torch.cat(
        [
            cells_batch.detach().reshape(n_systems, 9),
            periodic_batch.to(dtype).reshape(n_systems, 3),
        ],
        dim=1,
    ).cpu()
    cells_host = packed[:, :9].reshape(n_systems, 3, 3)
    periodic_host = packed[:, 9:] > 0.5

    n_periodic_axes: list[int] = []
    for b in range(n_systems):
        n_periodic_axes.append(int(periodic_host[b].sum()))
    if any(n == 1 for n in n_periodic_axes):
        raise ValueError("1D-periodic systems are not supported")
    is_periodic: list[bool] = []
    for n in n_periodic_axes:
        is_periodic.append(n > 0)

    # --- effective (2D-shrunk) cells and per-system Ewald parameters ---
    # only 2D slabs have a cell to shrink, and they are the one place where the
    # collation looks at the positions; they are shrunk on the device and copied back
    # in one batch, while a batch without slabs keeps the cells copied above as they are
    slab_systems: list[int] = []
    for b in range(n_systems):
        if n_periodic_axes[b] == 2:
            slab_systems.append(b)

    effective_cells_host = cells_host.clone()
    if len(slab_systems) > 0:
        shrunk: list[torch.Tensor] = []
        for b in slab_systems:
            shrunk.append(shrink_2d_cell(cells[b], periodic[b], positions[b]))
        effective_cells_host[torch.tensor(slab_systems, dtype=torch.long)] = (
            torch.stack(shrunk).detach().cpu()
        )

    lam_host, sigma_host, _ = ewald_params_from_num_k(
        effective_cells_host, periodic_host, num_k, smearing_factor=smearing_factor
    )
    if smearing is not None:
        override = smearing.to(sigma_host)
        sigma_host = torch.where(
            periodic_host.any(dim=-1), override.expand(n_systems), sigma_host
        )
    sigma = sigma_host.to(device=device, dtype=dtype)

    # --- concatenated atom layout ---
    n_atoms = [pos.shape[0] for pos in positions]
    atom_offsets = [0]
    for n in n_atoms:
        atom_offsets.append(atom_offsets[-1] + n)
    # the index arithmetic below stays on the host; only the collated tensors are
    # created on (or moved to) the device of the inputs
    system_index = torch.repeat_interleave(
        torch.arange(n_systems), torch.tensor(n_atoms)
    )

    # --- pair layout: screened (periodic systems) first, bare (non-periodic) after ---
    screened_indices, screened_distances, screened_sigma = [], [], []
    bare_indices, bare_distances = [], []
    for b in range(n_systems):
        offset_pairs = neighbor_indices[b] + atom_offsets[b]
        if is_periodic[b]:
            screened_indices.append(offset_pairs)
            screened_distances.append(neighbor_distances[b])
            screened_sigma.append(sigma[b].expand(neighbor_indices[b].shape[0]))
        else:
            bare_indices.append(offset_pairs)
            bare_distances.append(neighbor_distances[b])
    empty_indices = torch.empty(0, 2, dtype=torch.long, device=device)
    empty_distances = torch.empty(0, dtype=dtype, device=device)
    all_indices = torch.cat(screened_indices + bare_indices + [empty_indices])
    all_distances = torch.cat(screened_distances + bare_distances + [empty_distances])
    sigma_pair = torch.cat(screened_sigma + [empty_distances])
    n_screened_pairs = sigma_pair.shape[0]

    # --- per-system integer k-grids, padded to a common K_pad ---
    pbc_system: list[int] = []
    for b in range(n_systems):
        if is_periodic[b]:
            pbc_system.append(b)
    pbc_index = torch.tensor(pbc_system, dtype=torch.long)
    lengths_host = torch.linalg.norm(effective_cells_host, dim=-1)
    k_grids: list[torch.Tensor] = []
    for b in pbc_system:
        ns = [
            max(1, int(torch.ceil(lengths_host[b, i] / lam_host[b]))) for i in range(3)
        ]
        k_grids.append(_integer_kgrid(ns, halfspace))

    # pad the k-axis to at least (1 + k_pad_fraction)·num_k so that batches sharing
    # the same num_k get the same K_pad; a system whose per-axis rounding overshoots
    # the window still fits (K_pad grows), but shape stability across batches is lost
    k_floor = int((1.0 + k_pad_fraction) * num_k * (1 if halfspace else 2))
    k_realized = max([0] + [g.shape[0] for g in k_grids])
    if k_realized > k_floor:
        warnings.warn(
            f"A system realizes {k_realized} k-vectors, above the padding window "
            f"of {k_floor} for num_k={num_k}; the padded k-count grows accordingly. "
            "Increase `k_pad_fraction` to keep identical shapes across batches.",
            stacklevel=2,
        )
    k_pad = -(-max(k_floor, k_realized) // block_kvecs) * block_kvecs
    n_pbc = len(pbc_system)
    k_int = torch.zeros(n_pbc, k_pad, 3, dtype=torch.long)
    for row, grid in enumerate(k_grids):
        k_int[row, : grid.shape[0]] = grid

    # --- flat sum-padded atom layout over the periodic systems ---
    n_padded = [-(-n_atoms[b] // block_atoms) * block_atoms for b in pbc_system]
    pbc_atom_off = [0]
    for n in n_padded:
        pbc_atom_off.append(pbc_atom_off[-1] + n)
    n_flat = pbc_atom_off[-1]

    flat_to_atom = torch.zeros(n_flat, dtype=torch.long)
    flat_mask = torch.zeros(n_flat, dtype=torch.bool)
    for row, b in enumerate(pbc_system):
        start = pbc_atom_off[row]
        flat_to_atom[start : start + n_atoms[b]] = torch.arange(
            atom_offsets[b], atom_offsets[b] + n_atoms[b]
        )
        flat_mask[start : start + n_atoms[b]] = True

    # --- dispatch table: on-diagonal (system, atom_tile, k_tile) triples ---
    # outer atom-tile / inner k-tile order is load-bearing: it makes each
    # (system, atom_tile) group contiguous, so the kernel's second pass can reduce
    # with a reshape instead of a scatter
    n_kvec_tiles = k_pad // block_kvecs
    n_atom_tiles = torch.tensor([n // block_atoms for n in n_padded], dtype=torch.long)
    counts = n_atom_tiles * n_kvec_tiles
    n_triples = int(counts.sum())
    cum = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    b_col = torch.repeat_interleave(torch.arange(n_pbc), counts)
    inner = torch.arange(n_triples) - torch.repeat_interleave(cum, counts)
    mt_col = inner // n_kvec_tiles
    kt_col = inner % n_kvec_tiles

    flat_slots = (
        torch.tensor(pbc_atom_off[:-1], dtype=torch.long)[b_col] + mt_col * block_atoms
    ).unsqueeze(-1) + torch.arange(block_atoms)
    atom_gather = flat_to_atom[flat_slots]
    gather_mask = flat_mask[flat_slots]

    batch = {
        "positions": torch.cat(list(positions)),
        "charges": torch.cat(list(charges)),
        "cell": cells_batch,
        "periodic": periodic_batch,
        "system_index": system_index.to(device),
        "neighbor_indices": all_indices,
        "neighbor_distances": all_distances,
    }
    tiling = {
        "k_int": k_int,
        "sigma": sigma,
        "sigma_pair": sigma_pair,
        "pbc_system": pbc_index,
        "periodic_rows": periodic_host[pbc_index],
        "effective_cell_static": effective_cells_host[pbc_index],
        "b_col": b_col,
        "kt_col": kt_col,
        "atom_gather": atom_gather,
        "gather_mask": gather_mask,
        "flat_to_atom": flat_to_atom,
        "flat_mask": flat_mask,
        "pbc_atom": torch.tensor(is_periodic, dtype=torch.bool)[system_index],
        "block_atoms": torch.tensor(block_atoms, dtype=torch.long),
        "block_kvecs": torch.tensor(block_kvecs, dtype=torch.long),
        "n_kvec_tiles": torch.tensor(n_kvec_tiles, dtype=torch.long),
        "n_screened_pairs": torch.tensor(n_screened_pairs, dtype=torch.long),
        "g_factor": torch.tensor(2.0 if halfspace else 1.0, dtype=dtype),
    }
    return batch, {key: value.to(device) for key, value in tiling.items()}
