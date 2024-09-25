from typing import Dict, Literal, Optional, Tuple, Union
from warnings import warn

import torch
from torch.nn import ReLU

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
    def tune_ewald(
        cls,
        method: Union[Literal["fast", "medium", "accurate"], None],
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        subtract_interior: bool = False,
        accuracy: Optional[float] = None,
        exponent: float = 1.0,
        max_steps: int = 50000,
        learning_rate: float = 5e-2,
        verbose: bool = False,
    ) -> Tuple["EwaldPotential", torch.FloatTensor]:
        r"""Initialize based on a desired accuracy.

        :param optimize: the mode used to determine the optimal parameters. The
            possible values are ``"fast"``, ``"medium"``, ``"accurate"``.
            Under ``"fast"`` the parameters are set based on the number of
            atoms in the system to achieve a scaling of :math:`\mathcal{O}(N^{3/2})`.
            Under ``"medium"`` and ``"accurate"``, the parameters are optimized
            by using gradient descent until the estimated error is less than `accuracy`.
            The accuracy is set to 1e-3 and 1e-6 for``"medium"`` and
            ``"accurate"`` respectively. If `accuracy` is set, `optimize` is ignored.
        :param accuracy: the accuracy that should be achieved.
        :param positions: single tensor of shape (``len(charges), 3``) containing
            the Cartesian positions of all point charges in the system.
        :param charges: single tensor of shape (``1, len(positions))``.
        :param cell: single tensor of shape (3, 3), describing the bounding
            box/unit cell of the system. Each row should be one of the bounding box
            vector; and columns should contain the x, y, and z components of these
            vectors (i.e. the cell should be given in row-major order).
        :param subtract_interior: If set to :py:obj:`True`, subtract from the features
            of an atom the contributions to the potential arising from all atoms
            within the cutoff
            Note that if set to true, the self contribution (see previous) is also
            subtracted by default.
        :param max_steps: maximum number of gradient descent steps
        :param learning_rate: learning rate for gradient descent
        :param verbose: whether to print the progress of gradient descent

        :return: Tuple containing the initialized object and the optimal cutoff value
            for the neighborlist computation.

        Example
        -------
        Here we show how to use the ``from_accuracy`` method by using the example of a
        CsCl (Cesium-Chloride) crystal. The reference value of the energy is
        :math:`2 \cdot 1.7626 / \sqrt{3} \approx 2.0354`. For a detailed explanation,
        please refer to :py:class:`PMEPotential`.

        >>> import torch
        >>> from vesin.torch import NeighborList

        >>> positions = torch.tensor(
        ...     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
        ... )
        >>> charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        >>> cell = torch.eye(3, dtype=torch.float64)

        The ``from_accuracy`` method returns the initialized object and the optimal
        cutoff for the neighborlist computation.

        >>> calculator, cutoff = EwaldPotential.tune_ewald(
        ...     "medium", positions, charges, cell
        ... )

        You can check the values of the parameters

        >>> print(calculator.atomic_smearing)
        tensor(0.1676)

        >>> print(calculator.lr_wavelength)
        tensor(0.0732)

        >>> print(cutoff)
        tensor(0.6907)

        >>> nl = NeighborList(cutoff=cutoff, full_list=False)
        >>> i, j, neighbor_distances = nl.compute(
        ...     points=positions, box=cell, periodic=True, quantities="ijd"
        ... )
        >>> neighbor_indices = torch.stack([i, j], dim=1)
        >>> potentials = calculator.forward(
        ...     positions, charges, cell, neighbor_indices, neighbor_distances
        ... )
        >>> print(-torch.sum(potentials * charges))
        tensor(2.0354, dtype=torch.float64)

        Which is close to the reference value given above.

        Also you can set your own `accuracy`

        >>> calculator, cutoff = EwaldPotential.tune_ewald(
        ...     None, positions, charges, cell, accuracy=1e-3
        ... )

        Since the corresponding ``accuracy`` of the ``"medium"`` mode is 1e-3,
        this should return the same result.

        >>> print(calculator.atomic_smearing)
        tensor(0.1676)

        >>> print(calculator.lr_wavelength)
        tensor(0.0732)

        >>> print(cutoff)
        tensor(0.6907)
        """

        if exponent != 1:
            raise NotImplementedError("Only exponent = 1 is supported")

        device = positions.device
        # Create valid dummy values to verify `positions`, `charges` and `cell`
        neighbor_indices = torch.tensor([[0, 1], [1, 2]], device=device)
        neighbor_distances = torch.tensor([1, 2], device=device)
        (
            positions,
            charges,
            cell,
            _,
            _,
        ) = cls._validate_compute_parameters(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )

        if accuracy is not None and method is None:
            pass
        elif accuracy is not None and method is not None:
            warn("`optimize` is ignored if `accuracy` is set")
        elif method == "fast":
            smearing = torch.tensor(len(positions) ** (1 / 6) / 2**0.5)
            lr_wavelength = 2 * torch.pi * smearing
            cutoff = smearing
            ewald_potential = cls(
                atomic_smearing=smearing,
                lr_wavelength=lr_wavelength,
                subtract_interior=subtract_interior,
            )
            return ewald_potential, cutoff
        elif (method == "medium") and (accuracy is None):
            accuracy = 1e-3
        elif (method == "accurate") and (accuracy is None):
            accuracy = 1e-6
        elif method is None:
            raise ValueError("Either `optimize` or `accuracy` must be set")
        else:
            raise ValueError("`optimize` must be one of 'fast', 'medium' or 'accurate'")

        # Compute optimal values for atomic_smearing, lr_wavelength, and cutoff
        params, cutoff = cls._compute_optimal_parameters(
            accuracy=accuracy,
            positions=positions[0],
            charges=charges[0],
            cell=cell[0],
            exponent=exponent
            max_steps=max_steps,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        # Initialize the EwaldPotential object with the computed parameters
        ewald_potential = cls(**params, subtract_interior=subtract_interior)

        return ewald_potential, cutoff

    @staticmethod
    def _compute_optimal_parameters(
        accuracy: float,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        exponent: int,
        max_steps: int,
        learning_rate: float,
        verbose: bool,
    ) -> Tuple[Dict[str, float], float]:

        device = positions[0].device
        cell_dimensions = torch.linalg.norm(cell, dim=1)
        half_cell = float(torch.min(cell_dimensions) / 2)
        volume = torch.abs(cell.det())
        n_atoms = torch.tensor(len(positions), device=device)

        # For the error formula, please refer https://www2.icp.uni-stuttgart.de/~icp/
        # mediawiki/images/4/4d/Script_Longrange_Interactions.pdf
        # Please be careful to the difference between the parameters in the paper and
        # ours:
        # alpha=1/(sqrt(2)*smearing)
        # K=2*pi/sqrt(lr_wavelength)
        # r_c=cutoff

        err_Fourier = (
            lambda smearing, lr_wavelength: (
                torch.sum(charges**2) / torch.sqrt(n_atoms)
            )
            * 2**0.5
            / smearing
            / torch.sqrt(2 * torch.pi**2 * volume / lr_wavelength**0.5)
            * torch.exp(-2 * torch.pi**2 * smearing**2 / lr_wavelength)
        )
        err_real = (
            lambda smearing, cutoff: (torch.sum(charges**2) / torch.sqrt(n_atoms))
            * 2
            / torch.sqrt(cutoff * volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

        smearing = torch.tensor(half_cell / 5, device=device, requires_grad=True)
        lr_wavelength = torch.tensor(half_cell, device=device, requires_grad=True)
        cutoff = torch.tensor(half_cell / 10, device=device, requires_grad=True)

        optimizer = torch.optim.Adam(
            [smearing, lr_wavelength, cutoff], lr=learning_rate
        )
        relu = ReLU()
        for step in range(max_steps):
            result = (
                err_Fourier(
                    relu(smearing),
                    (2 * torch.sigmoid(lr_wavelength) + torch.tensor(5e-2)),
                )
                ** 2
                + err_real(relu(smearing), relu(cutoff)) ** 2
            )
            if torch.isnan(result):
                raise ValueError(
                    "The value of the estimated error is now nan, consider"
                    "using a smaller learning rate."
                )
            result.backward()
            optimizer.step()
            optimizer.zero_grad()
            if verbose and (step % 100 == 0):
                print(f"{step}: {result}")
            if result <= accuracy**2:
                break
        if result > accuracy**2:
            warn(
                "The searching for the parameters is ended, but the error is {:.3e}, "
                "larger than the given accuracy {:.3e}. Consider increase max_step and"
                " rerun.".format(result.detach().float() ** 0.5, accuracy)
            )

        return {
            "atomic_smearing": relu(smearing).detach().float(),
            "lr_wavelength": (2 * torch.sigmoid(lr_wavelength) + torch.tensor(5e-2))
            .detach()
            .float(),
        }, relu(cutoff).detach().float()

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
