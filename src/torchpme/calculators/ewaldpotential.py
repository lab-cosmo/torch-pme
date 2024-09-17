from typing import List, Optional, Tuple, Union

import torch

from ..lib import generate_kvectors_for_ewald
from ..lib.potentials import gamma
from .base import CalculatorBaseTorch


class EwaldPotential(CalculatorBaseTorch):
    r"""
    Potential computed using the Ewald sum.

    Scaling as :math:`\mathcal{O}(N^2)` with respect to the number of particles
    :math:`N`.

    For computing a **neighborlist** a reasonable ``cutoff`` is half the length of
    the shortest cell vector, which can be for example computed according as

    .. code-block:: python

        cell_dimensions = torch.linalg.norm(cell, dim=1)
        cutoff = torch.min(cell_dimensions) / 2 - 1e-6

    This ensures a accuracy of the short range part of ``1e-5``.

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :py:obj:`False`, a "half" neighbor list is expected.
    :param atomic_smearing: Width of the atom-centered Gaussian used to split the
        Coulomb potential into the short- and long-range parts. A reasonable value for
        most systems is to set it to ``1/5`` times the neighbor list cutoff. If
        :py:obj:`None` ,it will be set to 1/5 times of half the largest box vector
        (separately for each structure).
    :param lr_wavelength: Spatial resolution used for the long-range (reciprocal space)
        part of the Ewald sum. More conretely, all Fourier space vectors with a
        wavelength >= this value will be kept. If not set to a global value, it will be
        set to half the atomic_smearing parameter to ensure convergence of the
        long-range part to a relative precision of 1e-5.
    :param subtract_interior: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from all atoms within the cutoff
        Note that if set to true, the self contribution (see previous) is also
        subtracted by default.

    For an **example** on the usage refer to :py:class:`PMEPotential`.
    """

    def __init__(
        self,
        exponent: float = 1.0,
        atomic_smearing: Union[float, torch.Tensor, None] = None,
        lr_wavelength: Optional[float] = None,
        subtract_interior: bool = False,
        full_neighbor_list: bool = False,
    ):
        super().__init__(
            exponent=exponent,
            smearing=atomic_smearing,
            full_neighbor_list=full_neighbor_list,
        )

        self.lr_wavelength = lr_wavelength
        self.subtract_interior = subtract_interior

        if atomic_smearing is not None and atomic_smearing <= 0:
            raise ValueError(f"`atomic_smearing` {atomic_smearing} has to be positive")
        self.atomic_smearing = atomic_smearing

    @classmethod
    def from_accuracy(
        cls,
        accuracy: float,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]],
        subtract_interior: bool = False,
        max_steps: int = 10000,
    ) -> Tuple["EwaldPotential", float]:
        """Initialize based on a desired accuracy.

        :return: Tuple containing the initialized object and the optimal cutoff value
            for the neighborlist computation.
        """

        # TODO create valid dummy values to verify `positions`, `charges` and `cell`
        neighbor_indices = torch.tensor(0)
        neighbor_shifts = torch.tensor(0)
        # (
        #     positions,
        #     charges,
        #     cell,
        #     _,
        #     _,
        # ) = cls._validate_compute_parameters(
        #     positions=positions,
        #     charges=charges,
        #     cell=cell,
        #     neighbor_indices=neighbor_indices,
        #     neighbor_shifts=neighbor_shifts,
        # )
        # Compute optimal values for atomic_smearing, lr_wavelength, and cutoff
        atomic_smearing, lr_wavelength, cutoff = cls._compute_optimal_parameters(
            accuracy=accuracy,
            positions=positions,
            charges=charges,
            cell=cell,
            max_steps=max_steps,
        )

        # Initialize the EwaldPotential object with the computed parameters
        ewald_potential = cls(
            atomic_smearing=atomic_smearing,
            lr_wavelength=lr_wavelength,
            subtract_interior=subtract_interior,
        )

        return ewald_potential, cutoff

    @staticmethod
    def _compute_optimal_parameters(
        accuracy: float,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[Optional[torch.Tensor]], Optional[torch.Tensor]],
        max_steps: int,
    ) -> Tuple[float, float, float]:

        device = positions[0].device
        cell_dimensions = torch.linalg.norm(cell, dim=1)
        max_range = torch.min(cell_dimensions) / 2 - 1e-6
        G = torch.tensor(
            [
                [cell[0] @ cell[0], cell[0] @ cell[1], cell[0] @ cell[2]],
                [cell[1] @ cell[0], cell[1] @ cell[1], cell[1] @ cell[2]],
                [cell[2] @ cell[0], cell[2] @ cell[1], cell[2] @ cell[2]],
            ]
        )
        volume = torch.linalg.det(G) ** 0.5
        n_atoms = torch.tensor(len(positions), device=device)

        lr_wavelength = (
            torch.rand(1, device=device).uniform_(0, max_range).requires_grad_(True)
        )
        atomic_smearing = (
            torch.rand(1, device=device).uniform_(0, max_range).requires_grad_(True)
        )
        cutoff = (
            torch.rand(1, device=device).uniform_(0, max_range).requires_grad_(True)
        )

        # Step1: Find optimal lr_wavelength and atomic_smearing
        real_optimizer = torch.optim.Adam([atomic_smearing, cutoff], lr=0.1)
        err_real = (
            lambda atomic_smearing, cutoff: (torch.sum(charges**2) / torch.sqrt(n_atoms))
            * 2
            / torch.sqrt(cutoff * volume)
            * torch.exp(-(atomic_smearing**2) * cutoff**2)
        )

        steps = 0
        while ((result := err_real(atomic_smearing, torch.sigmoid(cutoff) * max_range)) > accuracy) and (
            steps < max_steps
        ):
            print(steps, result)
            result.backward()
            real_optimizer.step()
            real_optimizer.zero_grad()
            steps += 1
        atomic_smearing = atomic_smearing.detach()
        cutoff = torch.sigmoid(cutoff).detach() * max_range

        fourier_optimizer = torch.optim.Adam([lr_wavelength], lr=0.1)
        err_Fourier = (
            lambda lr_wavelength: (
                torch.sum(charges**2) / torch.sqrt(n_atoms)
            )
            * 2
            * atomic_smearing
            / torch.sqrt(torch.pi * lr_wavelength * volume)
            * torch.exp(-(lr_wavelength**2) / (4 * atomic_smearing**2))
        )

        steps = 0
        while ((result := err_Fourier(lr_wavelength)) > accuracy) and (
            steps < max_steps
        ):
            print(steps, result)
            result.backward()
            fourier_optimizer.step()
            fourier_optimizer.zero_grad()
            steps += 1
        lr_wavelength = lr_wavelength.detach()

        return atomic_smearing, lr_wavelength, cutoff

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        # Set the defaut values of convergence parameters
        # The total computational cost = cost of SR part + cost of LR part
        # Bigger smearing increases the cost of the SR part while decreasing the cost
        # of the LR part. Since the latter usually is more expensive, we maximize the
        # value of the smearing by default to minimize the cost of the LR part.
        # The auxilary parameter lr_wavelength then control the
        # convergence of the SR and LR sums, respectively. The default values are
        # chosen to reach a convergence on the order of 1e-4 to 1e-5 for the test
        # structures.
        if self.atomic_smearing is None:
            smearing = self.estimate_smearing(cell)
        else:
            smearing = self.atomic_smearing

        # TODO streamline the flow of parameters
        if self.lr_wavelength is None:
            lr_wavelength = 0.5 * smearing
        else:
            lr_wavelength = self.lr_wavelength

        # Compute short-range (SR) part using a real space sum
        potential_sr = self._compute_sr(
            is_periodic=True,
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            subtract_interior=self.subtract_interior,
        )

        # Compute long-range (LR) part using a Fourier / reciprocal space sum
        potential_lr = self._compute_lr(
            positions=positions,
            charges=charges,
            cell=cell,
            smearing=smearing,
            lr_wavelength=lr_wavelength,
        )

        return potential_sr + potential_lr

    def _compute_lr(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        smearing: float,
        lr_wavelength: float,
    ) -> torch.Tensor:
        # Define k-space cutoff from required real-space resolution
        k_cutoff = 2 * torch.pi / lr_wavelength

        # Compute number of times each basis vector of the reciprocal space can be
        # scaled until the cutoff is reached
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_float = k_cutoff * basis_norms / 2 / torch.pi
        ns = torch.ceil(ns_float).long()

        # Generate k-vectors and evaluate
        # kvectors = self._generate_kvectors(ns=ns, cell=cell)
        kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell)
        knorm_sq = torch.sum(kvectors**2, dim=1)

        # G(k) is the Fourier transform of the Coulomb potential
        # generated by a Gaussian charge density
        # We remove the singularity at k=0 by explicitly setting its
        # value to be equal to zero. This mathematically corresponds
        # to the requirement that the net charge of the cell is zero.
        # G = 4 * torch.pi * torch.exp(-0.5 * smearing**2 * knorm_sq) / knorm_sq
        G = self.potential.from_k_sq(knorm_sq)

        # Compute the energy using the explicit method that
        # follows directly from the Poisson summation formula.
        # For this, we precompute trigonometric factors for optimization, which leads
        # to N^2 rather than N^3 scaling.
        trig_args = kvectors @ (positions.T)  # shape num_k x num_atoms

        # Reshape charges into suitable form for array/tensor broadcasting
        num_atoms = len(positions)
        if charges.dim() > 1:
            num_channels = charges.shape[1]
            charges_reshaped = (charges.T).reshape(num_channels, 1, num_atoms)
            sum_idx = 2
        else:
            charges_reshaped = charges
            sum_idx = 1

        # Actual computation of trigonometric factors
        cos_all = torch.cos(trig_args)
        sin_all = torch.sin(trig_args)
        cos_summed = torch.sum(cos_all * charges_reshaped, dim=sum_idx)
        sin_summed = torch.sum(sin_all * charges_reshaped, dim=sum_idx)

        # Add up the contributions to compute the potential
        energy = torch.zeros_like(charges)
        for i in range(num_atoms):
            energy[i] += torch.sum(
                G * cos_all[:, i] * cos_summed, dim=sum_idx - 1
            ) + torch.sum(G * sin_all[:, i] * sin_summed, dim=sum_idx - 1)
        energy /= torch.abs(cell.det())

        # Remove the self-contribution: Using the Coulomb potential as an
        # example, this is the potential generated at the origin by the fictituous
        # Gaussian charge density in order to split the potential into a SR and LR part.
        # This contribution always should be subtracted since it depends on the smearing
        # parameter, which is purely a convergence parameter.
        phalf = self.exponent / 2
        fill_value = 1 / gamma(torch.tensor(phalf + 1)) / (2 * smearing**2) ** phalf
        self_contrib = torch.full([], fill_value)
        energy -= charges * self_contrib

        # Step 5: The method requires that the unit cell is charge-neutral.
        # If the cell has a net charge (i.e. if sum(charges) != 0), the method
        # implicitly assumes that a homogeneous background charge of the opposite sign
        # is present to make the cell neutral. In this case, the potential has to be
        # adjusted to compensate for this.
        # An extra factor of 2 is added to compensate for the division by 2 later on
        ivolume = torch.abs(cell.det()).pow(-1)
        charge_tot = torch.sum(charges, dim=0)
        prefac = torch.pi**1.5 * (2 * smearing**2) ** ((3 - self.exponent) / 2)
        prefac /= (3 - self.exponent) * gamma(torch.tensor(self.exponent / 2))
        energy -= 2 * prefac * charge_tot * ivolume

        # Compensate for double counting of pairs (i,j) and (j,i)
        return energy / 2
