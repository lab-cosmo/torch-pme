import torch

from .potential import Potential


def _pbc_correction(
    periodic: torch.Tensor | None,
    positions: torch.Tensor,
    cell: torch.Tensor,
    charges: torch.Tensor,
) -> torch.Tensor:
    # Define this helper function as this function is used in multiple potentials

    # "2D periodicity" correction for 1/r potential
    if periodic is None:
        periodic = torch.tensor([True, True, True], device=cell.device)
    n_periodic = torch.sum(periodic)
    is_2d = n_periodic == 2
    axis = torch.argmax(
        torch.where(
            is_2d.unsqueeze(-1),
            (~periodic).to(torch.int64),
            torch.zeros_like(periodic, dtype=torch.int64),
        ),
        dim=-1,
    )
    E_slab = torch.zeros_like(charges)
    z_i = torch.gather(positions, 1, axis.expand(positions.shape[0]).unsqueeze(-1))
    basis_len = torch.gather(torch.linalg.norm(cell, dim=-1), 0, axis)
    V = torch.abs(torch.linalg.det(cell))
    charge_tot = torch.sum(charges, dim=0)
    M_axis = torch.sum(charges * z_i, dim=0)
    M_axis_sq = torch.sum(charges * z_i**2, dim=0)
    E_slab_2d = (4.0 * torch.pi / V) * (
        z_i * M_axis
        - 0.5 * (M_axis_sq + charge_tot * z_i**2)
        - charge_tot / 12.0 * basis_len**2
    )

    return torch.where(is_2d.unsqueeze(-1), E_slab_2d, E_slab)


def _pbc_correction_batched(
    periodic: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    charges: torch.Tensor,
    system_index: torch.Tensor,
) -> torch.Tensor:
    # Batched, triclinic-correct 2D slab (Yeh-Berkowitz) correction for the 1/r
    # potential, evaluated on a concatenated batch of systems. The slab normal is the
    # cross product of the two periodic lattice vectors and heights are projections
    # onto it, which keeps the correction valid for non-orthorhombic cells.
    n_systems = cell.shape[0]
    is_2d = periodic.sum(dim=-1) == 2

    # index of the non-periodic axis (arbitrary but gated for non-2D systems)
    axis = torch.argmax((~periodic).to(torch.int64), dim=-1)
    rows = torch.arange(n_systems, device=cell.device)
    r1 = cell[rows, (axis + 1) % 3]
    r2 = cell[rows, (axis + 2) % 3]
    normal = torch.linalg.cross(r1, r2)
    normal_len = torch.linalg.norm(normal, dim=-1)
    n_hat = normal / normal_len.clamp(min=1e-15).unsqueeze(-1)
    # cell extent projected onto the slab normal
    basis_len = torch.abs((cell[rows, axis] * n_hat).sum(dim=-1))

    V = torch.abs(torch.linalg.det(cell))
    V = torch.where(is_2d, V, torch.ones_like(V))  # avoid 0-division on gated systems

    z_i = (positions * n_hat[system_index]).sum(dim=-1, keepdim=True)
    charge_tot = torch.zeros(
        n_systems, charges.shape[1], dtype=charges.dtype, device=charges.device
    ).index_add_(0, system_index, charges)
    M_axis = torch.zeros_like(charge_tot).index_add_(0, system_index, charges * z_i)
    M_axis_sq = torch.zeros_like(charge_tot).index_add_(
        0, system_index, charges * z_i**2
    )

    E_slab_2d = (4.0 * torch.pi / V[system_index]).unsqueeze(-1) * (
        z_i * M_axis[system_index]
        - 0.5 * (M_axis_sq[system_index] + charge_tot[system_index] * z_i**2)
        - charge_tot[system_index] / 12.0 * basis_len[system_index].unsqueeze(-1) ** 2
    )

    return torch.where(
        is_2d[system_index].unsqueeze(-1), E_slab_2d, torch.zeros_like(charges)
    )


class CoulombPotential(Potential):
    """
    Smoothed electrostatic Coulomb potential :math:`1/r`.

    Here :math:`r` is the inter-particle distance

    It can be used to compute:

    1. the full :math:`1/r` potential
    2. its short-range (SR) and long-range (LR) parts, the split being determined by a
       length-scale parameter (called "Inverse" in the code)
    3. the Fourier transform of the LR part

    :param smearing: float or torch.Tensor containing the parameter often called "sigma"
        in publications, which determines the length-scale at which the short-range and
        long-range parts of the naive :math:`1/r` potential are separated. The smearing
        parameter corresponds to the "width" of a Gaussian smearing of the particle
        density.
    :param exclusion_radius: A length scale that defines a *local environment* within
        which the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param exclusion_degree: Controls the sharpness of the transition in the cutoff function
        applied within the ``exclusion_radius``. The cutoff is computed as a raised cosine
        with exponent ``exclusion_degree``
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
    """

    def __init__(
        self,
        smearing: float | None = None,
        exclusion_radius: float | None = None,
        exclusion_degree: int = 1,
        prefactor: float = 1.0,
    ):
        super().__init__(smearing, exclusion_radius, exclusion_degree, prefactor)

    def from_dist(
        self, dist: torch.Tensor, pair_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Full :math:`1/r` potential as a function of :math:`r`.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        :param pair_mask: Optional torch.tensor containing a mask to be applied to the
            result.
        """
        result = 1.0 / dist.clamp(min=1e-15)

        if pair_mask is not None:
            result = result * pair_mask  # elementwise multiply, keeps shape fixed

        return self.prefactor * result

    def lr_from_dist(
        self,
        dist: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        smearing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Long range of the range-separated :math:`1/r` potential.

        Used to subtract out the interior contributions after computing the LR part in
        reciprocal (Fourier) space.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        :param pair_mask: Optional torch.tensor containing a mask to be applied to the
            result.
        :param smearing: Optional torch.tensor overriding the ``smearing`` buffer,
            broadcastable against ``dist``.
        """
        if smearing is None:
            smearing = self.smearing
        if smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )
        result = torch.erf(dist / smearing / 2.0**0.5) / dist.clamp(min=1e-12)
        if pair_mask is not None:
            result = result * pair_mask  # elementwise multiply, keeps shape fixed

        return self.prefactor * result

    def lr_from_k_sq(
        self, k_sq: torch.Tensor, smearing: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""
        Fourier transform of the LR part potential in terms of :math:`\mathbf{k^2}`.

        :param k_sq: torch.tensor containing the squared lengths (2-norms) of the wave
            vectors k at which the Fourier-transformed potential is to be evaluated
        :param smearing: Optional torch.tensor overriding the ``smearing`` buffer,
            broadcastable against ``k_sq``.
        """
        if smearing is None:
            smearing = self.smearing
        if smearing is None:
            raise ValueError(
                "Cannot compute long-range kernel without specifying `smearing`."
            )

        # avoid NaNs in backward, see
        # https://github.com/jax-ml/jax/issues/1052
        # https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
        masked = torch.where(k_sq == 0, 1.0, k_sq)
        return self.prefactor * torch.where(
            k_sq == 0,
            0.0,
            4 * torch.pi * torch.exp(-0.5 * smearing**2 * masked) / masked,
        )

    def self_contribution(self, smearing: torch.Tensor | None = None) -> torch.Tensor:
        # self-correction for 1/r potential
        if smearing is None:
            smearing = self.smearing
        if smearing is None:
            raise ValueError(
                "Cannot compute self contribution without specifying `smearing`."
            )
        return self.prefactor * (2 / torch.pi) ** 0.5 / smearing

    def background_correction(
        self, smearing: torch.Tensor | None = None
    ) -> torch.Tensor:
        # "charge neutrality" correction for 1/r potential
        if smearing is None:
            smearing = self.smearing
        if smearing is None:
            raise ValueError(
                "Cannot compute background correction without specifying `smearing`."
            )
        return self.prefactor * torch.pi * smearing**2

    def pbc_correction(
        self,
        periodic: torch.Tensor | None,
        positions: torch.Tensor,
        cell: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        return self.prefactor * _pbc_correction(periodic, positions, cell, charges)

    def pbc_correction_batched(
        self,
        periodic: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        charges: torch.Tensor,
        system_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.prefactor * _pbc_correction_batched(
            periodic, positions, cell, charges, system_index
        )

    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
    pbc_correction.__doc__ = Potential.pbc_correction.__doc__
    pbc_correction_batched.__doc__ = Potential.pbc_correction_batched.__doc__
