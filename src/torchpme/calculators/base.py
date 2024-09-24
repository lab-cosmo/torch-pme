from typing import List, Tuple, Union

import torch

from ..lib import InversePowerLawPotential


class CalculatorBaseTorch(torch.nn.Module):
    """Base calculator for the torch interface.

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials
    :param smearing: smearing parameter of a range separated potential
    """

    def __init__(
        self,
        exponent: float,
        smearing: Union[float, torch.Tensor] = None,
        use_half_neighborlist: bool = True,
    ):
        super().__init__()
        # TorchScript requires to initialize all attributes in __init__
        self._device = torch.device("cpu")
        self._dtype = torch.float32

        if exponent < 0.0 or exponent > 3.0:
            raise ValueError(f"`exponent` p={exponent} has to satisfy 0 < p <= 3")
        else:
            self.exponent = exponent
            self.potential = InversePowerLawPotential(
                exponent=exponent, smearing=smearing
            )

        self.use_half_neighborlist = use_half_neighborlist

    def _compute_sr(
        self,
        is_periodic: bool,
        charges: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
        subtract_interior: bool,
    ) -> torch.Tensor:

        if is_periodic:
            # If the contribution from all atoms within the cutoff is to be subtracted
            # this short-range part will simply use -V_LR as the potential
            if subtract_interior:
                potentials_bare = -self.potential.lr_from_dist(neighbor_distances)
            # In the remaining cases, we simply use the usual V_SR to get the full
            # 1/r^p potential when combined with the long-range part implemented in
            # reciprocal space
            else:
                potentials_bare = self.potential.sr_from_dist(neighbor_distances)
        else:  # is_direct
            potentials_bare = self.potential.from_dist(neighbor_distances)

        atom_is = neighbor_indices[:, 0]
        atom_js = neighbor_indices[:, 1]

        contributions_is = charges[atom_js] * potentials_bare.unsqueeze(-1)
        contributions_js = charges[atom_is] * potentials_bare.unsqueeze(-1)

        potential = torch.zeros_like(charges)
        potential.index_add_(0, atom_is, contributions_is)
        # If we are using a half neighbor list, we need to add the contributions
        # from the "inverse" pairs (j, i) to the atoms i
        if self.use_half_neighborlist:
            potential.index_add_(0, atom_js, contributions_js)

        # Compensate for double counting of pairs (i,j) and (j,i)
        return potential / 2

    @staticmethod
    def estimate_smearing(
        cell: torch.Tensor,
    ) -> float:
        """Estimate the smearing for ewald calculators.

        :param cell: A 3x3 tensor representing the periodic system
        :returns: estimated smearing
        """
        if torch.equal(
            cell.det(), torch.full([], 0, dtype=cell.dtype, device=cell.device)
        ):
            raise ValueError(
                "provided `cell` has a determinant of 0 and therefore is not valid "
                "for periodic calculation"
            )

        cell_dimensions = torch.linalg.norm(cell, dim=1)
        max_cutoff = torch.min(cell_dimensions) / 2 - 1e-6

        return max_cutoff.item() / 5.0

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

    def forward(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor],
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor],
        neighbor_distances: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute potential for all provided "systems" stacked inside list.

        The computation is performed on the same ``device`` as ``dtype`` is the input is
        stored on. The ``dtype`` of the output tensors will be the same as the input.

        :param positions: Single or 2D tensor of shape (``len(charges), 3``) containing
            the Cartesian positions of all point charges in the system.
        :param charges: Single 2D tensor or list of 2D tensor of shape (``n_channels,
            len(positions))``. ``n_channels`` is the number of charge channels the
            potential should be calculated for a standard potential ``n_channels=1``. If
            more than one "channel" is provided multiple potentials for the same
            position but different are computed.
        :param cell: single or 2D tensor of shape (3, 3), describing the bounding
            box/unit cell of the system. Each row should be one of the bounding box
            vector; and columns should contain the x, y, and z components of these
            vectors (i.e. the cell should be given in row-major order).
        :param neighbor_indices: Single or list of 2D tensors of shape ``(n, 2)``, where
            ``n`` is the number of neighbors. The two columns correspond to the indices
            of a **half neighbor list** for the two atoms which are considered neighbors
            (e.g. within a cutoff distance) if ``use_half_neighborlist=True``.
            Otherwise, a full neighbor list is expected.
        :param neighbor_distances: single or list of 1D tensors containing the distance
            between the ``n`` pairs corresponding to a **half (or full) neighbor list**
            (see ``neighbor_indices``).
        :return: Single or list of torch tensors containing the potential(s) for all
            positions. Each tensor in the list is of shape ``(len(positions),
            len(charges))``, where If the inputs are only single tensors only a single
            torch tensor with the potentials is returned.
        """
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
