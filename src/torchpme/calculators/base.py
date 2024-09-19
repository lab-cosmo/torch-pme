from typing import List, Tuple, Union

import torch

from ..lib import InversePowerLawPotential


class CalculatorBaseTorch(torch.nn.Module):
    """Base calculator for the torch interface."""

    def __init__(self):
        super().__init__()
        # TorchScript requires to initialize all attributes in __init__
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    @staticmethod
    def _validate_compute_parameters(
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor],
        neighbor_distances: Union[List[torch.Tensor], torch.Tensor],
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:

        # check that all inputs are of the same type
        for item, item_name in (
            (charges, "charges"),
            (cell, "cell"),
            (neighbor_indices, "neighbor_indices"),
            (neighbor_distances, "neighbor_distances"),
        ):
            if isinstance(positions, list):
                if isinstance(item, torch.Tensor):
                    raise TypeError(
                        "Inconsistent parameter types. `positions` is a "
                        f"list, while `{item_name}` is a torch.Tensor. Both need "
                        "either be a list or a torch.Tensor!"
                    )
            else:
                if isinstance(item, list):
                    raise TypeError(
                        "Inconsistent parameter types. `positions` is a "
                        f"torch.Tensor, while `{item_name}` is a list. Both need "
                        "either be a list or a torch.Tensor!"
                    )

        # make sure that all provided parameters are lists
        if not isinstance(positions, list):
            positions = [positions]

        if not isinstance(charges, list):
            charges = [charges]

        if not isinstance(cell, list):
            cell = [cell]

        if not isinstance(neighbor_indices, list):
            neighbor_indices = [neighbor_indices]

        if not isinstance(neighbor_distances, list):
            neighbor_distances = [neighbor_distances]

        device = positions[0].device
        dtype = positions[0].dtype

        for item, item_name in (
            (charges, "charges"),
            (cell, "cell"),
            (neighbor_indices, "neighbor_indices"),
            (neighbor_distances, "neighbor_distances"),
        ):
            if len(positions) != len(item):
                raise ValueError(
                    f"Got inconsistent numbers of positions ({len(positions)}) and "
                    f"{item_name} ({len(item)})"
                )

        # check that all devices and data types (dtypes) are consistent
        for (
            positions_single,
            cell_single,
            charges_single,
            neighbor_indices_single,
            neighbor_distances_single,
        ) in zip(positions, cell, charges, neighbor_indices, neighbor_distances):
            # check shape, dtype and device of positions
            num_atoms = len(positions_single)
            if list(positions_single.shape) != [num_atoms, 3]:
                raise ValueError(
                    "each `positions` must be a tensor with shape [n_atoms, 3], got at "
                    f"least one tensor with shape {list(positions_single.shape)}"
                )

            if positions_single.dtype != dtype:
                raise ValueError(
                    f"each `positions` must have the same type {dtype} as the "
                    "first provided one. Got at least one tensor of type "
                    f"{positions_single.dtype}"
                )

            if positions_single.device != device:
                raise ValueError(
                    f"each `positions` must be on the same device {device} as "
                    "the first provided one. Got at least one tensor on device "
                    f"{positions_single.device}"
                )

            # check shape, dtype and device of cell
            if list(cell_single.shape) != [3, 3]:
                raise ValueError(
                    "each `cell` must be a tensor with shape [3, 3], got at least "
                    f"one tensor with shape {list(cell_single.shape)}"
                )

            if cell_single.dtype != dtype:
                raise ValueError(
                    f"each `cell` must have the same type {dtype} as "
                    "`positions`, got at least one tensor of type "
                    f"{cell_single.dtype}"
                )

            if cell_single.device != device:
                raise ValueError(
                    f"each `cell` must be on the same device {device} as "
                    "`positions`, got at least one tensor with device "
                    f"{cell_single.device}"
                )

            # check shape, dtype & device of `charges`
            if charges_single.dim() != 2:
                raise ValueError(
                    "each `charges` needs to be a 2-dimensional tensor, got at least "
                    f"one tensor with {charges_single.dim()} dimension(s) and shape "
                    f"{list(charges_single.shape)}"
                )

            if list(charges_single.shape) != [num_atoms, charges_single.shape[1]]:
                raise ValueError(
                    "each `charges` must be a tensor with shape [n_atoms, n_channels], "
                    "with `n_atoms` being the same as the variable `positions`. Got at "
                    f"least one tensor with shape {list(charges_single.shape)} where "
                    f"positions contains {len(positions_single)} atoms"
                )

            if charges_single.dtype != dtype:
                raise ValueError(
                    f"each `charges` must have the same type {dtype} as "
                    "`positions`, got at least one tensor of type "
                    f"{charges_single.dtype}"
                )

            if charges_single.device != device:
                raise ValueError(
                    f"each `charges` must be on the same device {device} as "
                    f"`positions`, got at least one tensor with device "
                    f"{charges_single.device}"
                )

            # check shape, dtype & device of `neighbor_indices` and `neighbor_distances`
            if neighbor_indices_single.shape[1] != 2:
                raise ValueError(
                    "neighbor_indices is expected to have shape [num_neighbors, 2]"
                    f", but got {list(neighbor_indices_single.shape)} for one "
                    "structure"
                )

            if neighbor_indices_single.device != device:
                raise ValueError(
                    f"each `neighbor_indices` must be on the same device "
                    f"{device} as `positions`, got at least one tensor with "
                    f"device {neighbor_indices_single.device}"
                )

            if neighbor_distances_single.shape != neighbor_indices_single[:, 0].shape:
                raise ValueError(
                    "`neighbor_indices` and `neighbor_distances` need to have shapes "
                    "[num_neighbors, 2] and [num_neighbors]. For at least one "
                    f"structure, got {list(neighbor_indices_single.shape)} and "
                    f"{list(neighbor_distances_single.shape)}, "
                    "which is inconsistent"
                )

            if neighbor_distances_single.device != device:
                raise ValueError(
                    f"each `neighbor_distances` must be on the same device "
                    f"{device} as `positions`, got at least one tensor with "
                    f"device {neighbor_distances_single.device}"
                )

        return positions, charges, cell, neighbor_indices, neighbor_distances

    def _compute_impl(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor],
        neighbor_distances: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        # save if the inputs were lists or single tensors
        input_is_list = isinstance(positions, list)

        # Check that all shapes, data types and devices are consistent
        # Furthermore, to handle the special case in which only the inputs for a single
        # structure are provided, turn inputs into a list to be consistent with the
        # more general case
        (
            positions,
            charges,
            cell,
            neighbor_indices,
            neighbor_distances,
        ) = self._validate_compute_parameters(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )

        # In actual computations, the data type (dtype) and device (e.g. CPU, GPU) of
        # all remaining variables need to be consistent
        self._device = positions[0].device
        self._dtype = positions[0].dtype

        # compute and append into a list the features of each structure
        potentials = []
        for (
            positions_single,
            charges_single,
            cell_single,
            neighbor_indices_single,
            neighbor_distances_single,
        ) in zip(positions, charges, cell, neighbor_indices, neighbor_distances):
            # `_compute_single_system` is implemented only in child classes!
            potentials.append(
                self._compute_single_system(
                    positions=positions_single,
                    charges=charges_single,
                    cell=cell_single,
                    neighbor_indices=neighbor_indices_single,
                    neighbor_distances=neighbor_distances_single,
                )
            )

        if input_is_list:
            return potentials
        else:
            return potentials[0]


class PeriodicBase(torch.nn.Module):
    """Base class providing general funtionality for periodic calculations.

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials
    :param atomic_smearing: Width of the atom-centered Gaussian used to split the
        Coulomb potential into the short- and long-range parts. A reasonable value for
        most systems is to set it to ``1/5`` times the neighbor list cutoff. If
        :py:obj:`None` ,it will be set to 1/5 times of half the largest box vector
        (separately for each structure).
    :param subtract_interior: If set to :py:obj:`True`, subtract from the features of an
        atom the contributions to the potential arising from all atoms within the cutoff
        Note that if set to true, the self contribution (see previous) is also
        subtracted by default.
    """

    def __init__(
        self,
        exponent: float,
        atomic_smearing: Union[None, float],
        subtract_interior: bool,
    ):
        super().__init__()

        if exponent < 0.0 or exponent > 3.0:
            raise ValueError(f"`exponent` p={exponent} has to satisfy 0 < p <= 3")
        if atomic_smearing is not None and atomic_smearing <= 0:
            raise ValueError(f"`atomic_smearing` {atomic_smearing} has to be positive")

        # Attach the function handling all computations related to the
        # power-law potential for later convenience
        self.exponent = exponent
        self.atomic_smearing = atomic_smearing
        self.smearing = 1.0 if atomic_smearing is None else atomic_smearing
        self.subtract_interior = subtract_interior
        self.potential = InversePowerLawPotential(
            exponent=exponent, smearing=self.smearing
        )

    def _compute_sr(
        self,
        smearing: float,
        charges: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        # If the contribution from all atoms within the cutoff is to be subtracted
        # this short-range part will simply use -V_LR as the potential
        if self.subtract_interior:
            potentials_bare = -self.potential.from_dist_lr(neighbor_distances)
        # In the remaining cases, we simply use the usual V_SR to get the full
        # 1/r^p potential when combined with the long-range part implemented in
        # reciprocal space
        else:
            potentials_bare = self.potential.from_dist_sr(neighbor_distances)

        atom_is = neighbor_indices[:, 0]
        atom_js = neighbor_indices[:, 1]

        contributions_is = charges[atom_js] * potentials_bare.unsqueeze(-1)
        contributions_js = charges[atom_is] * potentials_bare.unsqueeze(-1)

        potential = torch.zeros_like(charges)
        potential.index_add_(0, atom_is, contributions_is)
        potential.index_add_(0, atom_js, contributions_js)

        return potential / 2

    def _estimate_smearing(
        self,
        cell: torch.Tensor,
    ) -> float:
        if torch.equal(
            cell.det(), torch.full([], 0, dtype=cell.dtype, device=cell.device)
        ):
            raise ValueError(
                "provided `cell` has a determinant of 0 and therefore is not valid "
                "for periodic calculation"
            )

        if self.atomic_smearing is None:
            cell_dimensions = torch.linalg.norm(cell, dim=1)
            max_cutoff = torch.min(cell_dimensions) / 2 - 1e-6
            smearing = max_cutoff.item() / 5.0
        else:
            smearing = self.atomic_smearing

        # also sets the internal state of the smearing
        self.smearing = smearing
        self.potential.smearing = smearing

        return smearing
