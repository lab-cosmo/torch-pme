import math
import warnings
from typing import Literal, Optional

import torch

from ..lib import Potential, generate_kvectors_for_ewald
from .base import Calculator, estimate_smearing


class EwaldCalculator(Calculator):
    r"""
    Potential computed using the Ewald sum.

    Scaling as :math:`\mathcal{O}(N^2)` with respect to the number of particles
    :math:`N`.

    For computing a **neighborlist** a reasonable ``cutoff`` is half the length of the
    shortest cell vector, which can be for example computed according as

    .. code-block:: python

        cell_dimensions = torch.linalg.norm(cell, dim=1)
        cutoff = torch.min(cell_dimensions) / 2 - 1e-6

    For an electrostatic potential, and following the guidelines discussed below
    for the other parameters, this ensures an accuracy of approximately ``1e-5``.

    :param potential: the two-body potential that should be computed, implemented
        as a :py:class:`torchpme.lib.Potential` object. The ``smearing`` parameter
        of the potential determines the split between real and k-space regions.
        For a :py:class:`torchpme.lib.CoulombPotential` it corresponds to the
        width of the atom-centered Gaussian used to split the Coulomb potential
        into the short- and long-range parts. A reasonable value for
        most systems is to set it to ``1/5`` times the neighbor list cutoff.
    :param lr_wavelength: Spatial resolution used for the long-range (reciprocal space)
        part of the Ewald sum. More concretely, all Fourier space vectors with a
        wavelength >= this value will be kept. If not set to a global value, it will be
        set to half the smearing parameter to ensure convergence of the
        long-range part to a relative precision of 1e-5.
    :param full_neighbor_list: If set to :py:obj:`True`, a "full" neighbor list is
        expected as input. This means that each atom pair appears twice. If set to
        :py:obj:`False`, a "half" neighbor list is expected.

    To tune the ``smearing`` and  ``lr_wavelength`` for a system you can use the
    :py:func:`tune_pme` function.

    For an **example** on the usage for any calculator refer to :ref:`userdoc-how-to`.
    """

    def __init__(
        self,
        potential: Potential,
        lr_wavelength: Optional[float] = None,
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
        self.lr_wavelength: float = lr_wavelength or potential.smearing * 0.5

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
        # kvectors = self._generate_kvectors(ns=ns, cell=cell)
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
        fill_value = self.potential.self_contribution()
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
        prefac = self.potential.background_correction()
        energy -= 2 * prefac * charge_tot * ivolume

        # Compensate for double counting of pairs (i,j) and (j,i)
        return energy / 2


def tune_ewald(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    exponent: int = 1,
    accuracy: Optional[Literal["fast", "medium", "accurate"] | float] = "fast",
    max_steps: int = 50000,
    learning_rate: float = 5e-2,
    verbose: bool = False,
) -> tuple[dict[str, float], float]:
    r"""Find the optimal parameters for a single system for the ewald method.

    For the error formulas are given `elsewhere <https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/4/4d/Script_Longrange_Interactions.pdf>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha &= \left( \sqrt{2}\,\mathrm{smearing} \right)^{-1}

        K &= \frac{2 \pi}{\mathrm{lr\_wavelength}}

        r_c &= \mathrm{cutoff}

    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    :param charges: single tensor of shape (``1, len(positions))``.
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param exponent: exponent :math:`p` in :math:`1/r^p` potentials
    :param accuracy: Mode used to determine the optimal parameters. Possible values are
        ``"fast"``, ``"medium"`` or ``"accurate"``. For ``"fast"`` the parameters are
        set based on the number of atoms in the system to achieve a scaling of
        :math:`\mathcal{O}(N^{3/2})`. For ``"medium"`` or ``"accurate"``, the parameters
        are optimized using gradient descent until an estimated error of :math:`10^{-3}`
        or :math:`10^{-6}` is reached.
        Instead of ``"fast"``, ``"medium"`` or ``"accurate"``, you can give a float
        value for the accuracy.
    :param max_steps: maximum number of gradient descent steps
    :param learning_rate: learning rate for gradient descent
    :param verbose: whether to print the progress of gradient descent

    :return: Tuple containing a dictionary with the parameters for
        :py:class:`CalculatorEwald` and a float of the optimal cutoff value for the
        neighborlist computation.

    Example
    -------
    >>> import torch
    >>> from vesin.torch import NeighborList
    >>> positions = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    ... )
    >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    >>> cell = torch.eye(3, dtype=torch.float64)
    >>> ewald_parameter, cutoff = tune_ewald(charges, cell, positions, accuracy="fast")

    You can check the values of the parameters

    >>> print(ewald_parameter)
    {'smearing': 1.0318106837793297, 'lr_wavelength': 2.9468444218696392}

    >>> print(cutoff)
    2.2699835043145256

    Which can be used to initilize an :py:class:`CalculatorEwald` instance with
    parameters that are optimal for the system.
    """

    if exponent != 1:
        raise NotImplementedError("Only exponent = 1 is supported")

    dtype = positions.dtype
    device = positions.device

    # Create valid dummy tensors to verify `positions`, `charges` and `cell`
    neighbor_indices = torch.zeros(0, 2, device=device)
    neighbor_distances = torch.zeros(0, device=device)
    Calculator._validate_compute_parameters(
        charges=charges,
        cell=cell,
        positions=positions,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )

    if charges.shape[1] > 1:
        raise NotImplementedError(
            f"Found {charges.shape[1]} charge channels, but only one iss supported"
        )

    if accuracy == "fast":
        # The factors below are chosen to achieve an additional improved balance
        # between accuracy and speed, while maintaining a N^3/2 scaling. The values
        # result from tests on a CsCl system, whose unit cell is repeated 16 times
        # in each direction, resulting in a system of 8192 atoms.
        smearing_factor = 1.3
        lr_wavelength_factor = 2.2

        smearing = smearing_factor * len(positions) ** (1 / 6) / 2**0.5

        return {
            "smearing": smearing,
            "lr_wavelength": 2 * torch.pi * smearing / lr_wavelength_factor,
        }, smearing * lr_wavelength_factor

    if accuracy == "medium":
        accuracy = 1e-3
    elif accuracy == "accurate":
        accuracy = 1e-6
    elif not isinstance(accuracy, float):
        raise ValueError(
            f"'{accuracy}' is not a valid method or a float: Choose from 'fast',"
            f"'medium' or 'accurate', or provide a float for the accuracy."
        )

    cell_dimensions = torch.linalg.norm(cell, dim=1)
    min_dimension = float(torch.min(cell_dimensions))
    half_cell = float(torch.min(cell_dimensions) / 2)

    smearing_init = estimate_smearing(cell)
    prefac = 2 * torch.sum(charges**2) / math.sqrt(len(positions))
    volume = torch.abs(cell.det())

    def smooth_lr_wavelength(lr_wavelength):
        """Confine to (0, min_dimension), ensuring that the ``ns``
        parameter is not smaller than 1
        (see :py:func:`_compute_lr` of :py:class:`CalculatorEwald`)."""
        return min_dimension * torch.sigmoid(lr_wavelength)

    def err_Fourier(smearing, lr_wavelength):
        return (
            prefac**0.5
            / smearing
            / torch.sqrt(2 * torch.pi**2 * volume / lr_wavelength**0.5)
            * torch.exp(-2 * torch.pi**2 * smearing**2 / lr_wavelength)
        )

    def err_real(smearing, cutoff):
        return (
            prefac
            / torch.sqrt(cutoff * volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def loss(smearing, lr_wavelength, cutoff):
        return torch.sqrt(
            err_Fourier(smearing, lr_wavelength) ** 2 + err_real(smearing, cutoff) ** 2
        )

    # initial guess
    smearing = torch.tensor(
        smearing_init, device=device, dtype=dtype, requires_grad=True
    )
    lr_wavelength = torch.tensor(
        half_cell, device=device, dtype=dtype, requires_grad=True
    )
    cutoff = torch.tensor(
        half_cell / 10, device=device, dtype=dtype, requires_grad=True
    )

    optimizer = torch.optim.Adam([smearing, lr_wavelength, cutoff], lr=learning_rate)

    for step in range(max_steps):
        loss_value = loss(smearing, smooth_lr_wavelength(lr_wavelength), cutoff)
        if torch.isnan(loss_value):
            raise ValueError(
                "The value of the estimated error is now nan, consider using a "
                "smaller learning rate."
            )
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose and (step % 100 == 0):
            print(f"{step}: {loss_value:.2e}")
        if loss_value <= accuracy:
            break

    if loss_value > accuracy:
        warnings.warn(
            "The searching for the parameters is ended, but the error is "
            f"{float(loss_value):.3e}, larger than the given accuracy {accuracy}. "
            "Consider increase max_step and",
            stacklevel=2,
        )

    return {
        "smearing": float(smearing),
        "lr_wavelength": float(smooth_lr_wavelength(lr_wavelength)),
    }, float(cutoff)
