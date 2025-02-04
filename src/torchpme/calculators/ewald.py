from typing import Optional, Union

import torch

from ..lib import generate_kvectors_for_ewald
from ..potentials import Potential
from .calculator import Calculator


class EwaldCalculator(Calculator):
    r"""
    Potential computed using the Ewald sum.

    Scaling as :math:`\mathcal{O}(N^2)` with respect to the number of particles
    :math:`N`.

    For getting reasonable values for the ``smaring`` of the potential class and  the
    ``lr_wavelength`` based on a given accuracy for a specific structure you should use
    :func:`torchpme.tuning.tune_ewald`. This function will also find the optimal
    ``cutoff`` for the  **neighborlist**.

    .. hint::

        For a training exercise it is recommended only run a tuning procedure with
        :func:`torchpme.tuning.tune_ewald` for the largest system in your dataset.

    For a :math:`\mathcal{O}(N^{1.5})` scaling, one can set the parameters as following:

    .. math::

        \mathrm{smearing} &= 1.3 \cdot N^{1 / 6} / \sqrt{2}

        \mathrm{lr\_wavelength} &= 2 \pi \cdot \mathrm{smearing} / 2.2

        \mathrm{r_c} &= 2.2 \cdot \mathrm{smearing}

    where :math:`N` is the number of particles. The magic numbers :math:`1.3` and
    :math:`2.2` above result from tests on a CsCl system, whose unit cell is repeated 16
    times in each direction, resulting in a system of 8192 atoms.

    :param potential: the two-body potential that should be computed, implemented
        as a :class:`torchpme.potentials.Potential` object. The ``smearing`` parameter
        of the potential determines the split between real and k-space regions.
        For a :class:`torchpme.CoulombPotential` it corresponds to the
        width of the atom-centered Gaussian used to split the Coulomb potential
        into the short- and long-range parts. A reasonable value for
        most systems is to set it to ``1/5`` times the neighbor list cutoff.
    :param lr_wavelength: Spatial resolution used for the long-range (reciprocal space)
        part of the Ewald sum. More concretely, all Fourier space vectors with a
        wavelength >= this value will be kept. If not set to a global value, it will be
        set to half the smearing parameter to ensure convergence of the
        long-range part to a relative precision of 1e-5.
    :param full_neighbor_list: If set to :obj:`True`, a "full" neighbor list is
        expected as input. This means that each atom pair appears twice. If set to
        :obj:`False`, a "half" neighbor list is expected.
    :param prefactor: electrostatics prefactor; see :ref:`prefactors` for details and
        common values.
    """

    def __init__(
        self,
        potential: Potential,
        lr_wavelength: float,
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
    ):
        super().__init__(
            potential=potential,
            full_neighbor_list=full_neighbor_list,
            prefactor=prefactor,
        )
        if potential.smearing is None:
            raise ValueError(
                "Must specify range radius to use a potential with EwaldCalculator"
            )
        self.lr_wavelength: float = lr_wavelength

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Define k-space cutoff from required real-space resolution
        k_cutoff = 2 * torch.pi / self.lr_wavelength

        # Compute number of times each basis vector of the reciprocal space can be
        # scaled until the cutoff is reached
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_float = k_cutoff * basis_norms / 2 / torch.pi
        ns = torch.ceil(ns_float).long()

        # Generate k-vectors and evaluate
        kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell)
        knorm_sq = torch.sum(kvectors**2, dim=1)

        # G(k) is the Fourier transform of the Coulomb potential
        # generated by a Gaussian charge density
        # We remove the singularity at k=0 by explicitly setting its
        # value to be equal to zero. This mathematically corresponds
        # to the requirement that the net charge of the cell is zero.
        # G = 4 * torch.pi * torch.exp(-0.5 * smearing**2 * knorm_sq) / knorm_sq
        G = self.potential.lr_from_k_sq(knorm_sq)

        # Compute the energy using the explicit method that
        # follows directly from the Poisson summation formula.
        # For this, we precompute trigonometric factors for optimization, which leads
        # to N^2 rather than N^3 scaling.
        trig_args = kvectors @ (positions.T)  # [k, i]

        c = torch.cos(trig_args)  # [k, i]
        s = torch.sin(trig_args)  # [k, i]
        sc = torch.stack([c, s], dim=0)  # [2 "f", k, i]
        sc_summed_G = torch.einsum("fki,ic, k->fkc", sc, charges, G)
        energy = torch.einsum("fkc,fki->ic", sc_summed_G, sc)
        energy /= torch.abs(cell.det())

        # Remove the self-contribution: Using the Coulomb potential as an
        # example, this is the potential generated at the origin by the fictituous
        # Gaussian charge density in order to split the potential into a SR and LR part.
        # This contribution always should be subtracted since it depends on the smearing
        # parameter, which is purely a convergence parameter.
        energy -= charges * self.potential.self_contribution()

        # Step 5: The method requires that the unit cell is charge-neutral.
        # If the cell has a net charge (i.e. if sum(charges) != 0), the method
        # implicitly assumes that a homogeneous background charge of the opposite sign
        # is present to make the cell neutral. In this case, the potential has to be
        # adjusted to compensate for this.
        # An extra factor of 2 is added to compensate for the division by 2 later on
        ivolume = torch.abs(cell.det()).pow(-1)
        charge_tot = torch.sum(charges, dim=0)
        prefac = self.potential.background_correction()
        energy -= 2 * prefac * charge_tot * ivolume
        # Compensate for double counting of pairs (i,j) and (j,i)
        return energy / 2
