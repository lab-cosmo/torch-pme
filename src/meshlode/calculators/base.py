from typing import List, Optional, Tuple, Union

import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from meshlode.lib import InversePowerLawPotential


class _ShortRange:
    """Base class providing general funtionality for short range interactions."""

    def __init__(self, exponent: float, subtract_interior: bool):
        # Attach the function handling all computations related to the
        # power-law potential for later convenience
        self.exponent = exponent
        self.subtract_interior = subtract_interior
        self.potential = InversePowerLawPotential(exponent=exponent)

    def _compute_sr(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        smearing: float,
        sr_cutoff: torch.Tensor,
        neighbor_indices: Optional[torch.Tensor] = None,
        neighbor_shifts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if neighbor_indices is None or neighbor_shifts is None:
            # Get list of neighbors
            struc = Atoms(positions=positions.detach().numpy(), cell=cell, pbc=True)
            atom_is, atom_js, neighbor_shifts = neighbor_list(
                "ijS", struc, sr_cutoff.item(), self_interaction=False
            )
            atom_is = torch.tensor(atom_is)
            atom_js = torch.tensor(atom_js)
            shifts = torch.tensor(neighbor_shifts, dtype=cell.dtype)  # N x 3
        else:
            atom_is = neighbor_indices[0]
            atom_js = neighbor_indices[1]
            shifts = neighbor_shifts.type(cell.dtype).T

        # Compute energy
        potential = torch.zeros_like(charges)

        pos_is = positions[atom_is]
        pos_js = positions[atom_js]
        dists = torch.linalg.norm(pos_js - pos_is + shifts @ cell, dim=1)
        # If the contribution from all atoms within the cutoff is to be subtracted
        # this short-range part will simply use -V_LR as the potential
        if self.subtract_interior:
            potentials_bare = -self.potential.potential_lr_from_dist(dists, smearing)
        # In the remaining cases, we simply use the usual V_SR to get the full
        # 1/r^p potential when combined with the long-range part implemented in
        # reciprocal space
        else:
            potentials_bare = self.potential.potential_sr_from_dist(dists, smearing)
        # potential.index_add_(0, atom_is, charges[atom_js] * potentials_bare)

        for i, j, potential_bare in zip(atom_is, atom_js, potentials_bare):
            potential[i.item()] += charges[j.item()] * potential_bare

        return potential


class CalculatorBaseTorch(torch.nn.Module):
    """
    Base calculator for the torch interface to MeshLODE.

    :param exponent: the exponent :math:`p` in :math:`1/r^p` potentials
    """

    def __init__(
        self,
    ):
        super().__init__()

    def _validate_compute_parameters(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[None, List[torch.Tensor], torch.Tensor],
        neighbor_indices: Union[None, List[torch.Tensor], torch.Tensor],
        neighbor_shifts: Union[None, List[torch.Tensor], torch.Tensor],
    ) -> Tuple[
        List[torch.Tensor],
        Union[List[None], List[torch.Tensor]],
        List[torch.Tensor],
        Union[List[None], List[torch.Tensor]],
        Union[List[None], List[torch.Tensor]],
    ]:
        # make sure that the provided positions are a list
        if not isinstance(positions, list):
            positions = [positions]

        # In actual computations, the data type (dtype) and device (e.g. CPU, GPU) of
        # all remaining variables need to be consistent
        self._device = positions[0].device
        self._dtype = positions[0].dtype

        # make sure that provided cells are a list of same length as positions
        if cell is None:
            cell = len(positions) * [None]
        elif not isinstance(cell, list):
            cell = [cell]

        if len(positions) != len(cell):
            raise ValueError(
                f"Got inconsistent numbers of positions ({len(positions)}) and "
                f"cell ({len(cell)})"
            )

        # make sure that provided charges are a list of same length as positions
        if not isinstance(charges, list):
            charges = [charges]

        if len(positions) != len(charges):
            raise ValueError(
                f"Got inconsistent numbers of positions ({len(positions)}) and "
                f"charges ({len(charges)})"
            )

        # check neighbor_indices
        if neighbor_indices is None:
            neighbor_indices = len(positions) * [None]
        elif not isinstance(neighbor_indices, list):
            neighbor_indices = [neighbor_indices]

        if len(positions) != len(neighbor_indices):
            raise ValueError(
                f"Got inconsistent numbers of positions ({len(positions)}) and "
                f"neighbor_indices ({len(neighbor_indices)})"
            )

        # check neighbor_shifts
        if neighbor_shifts is None:
            neighbor_shifts = len(positions) * [None]
        elif not isinstance(neighbor_shifts, list):
            neighbor_shifts = [neighbor_shifts]

        if len(positions) != len(neighbor_shifts):
            raise ValueError(
                f"Got inconsistent numbers of positions ({len(positions)}) and "
                f"neighbor_shifts ({len(neighbor_shifts)})"
            )

        # check that all devices and data types (dtypes) are consistent
        for (
            positions_single,
            cell_single,
            charges_single,
            neighbor_indices_single,
            neighbor_shifts_single,
        ) in zip(positions, cell, charges, neighbor_indices, neighbor_shifts):
            # check shape, dtype and device of positions
            num_atoms = len(positions_single)
            if list(positions_single.shape) != [num_atoms, 3]:
                raise ValueError(
                    "each `positions` must be a (n_atoms x 3) tensor, got at least "
                    f"one tensor with shape {tuple(positions_single.shape)}"
                )

            if positions_single.dtype != self._dtype:
                raise ValueError(
                    f"each `positions` must have the same type {self._dtype} as the "
                    "first provided one. Got at least one tensor of type "
                    f"{positions_single.dtype}"
                )

            if positions_single.device != self._device:
                raise ValueError(
                    f"each `positions` must be on the same device {self._device} as "
                    "the first provided one. Got at least one tensor on device "
                    f"{positions_single.device}"
                )

            # check shape, dtype and device of cell
            if cell_single is not None:
                if list(cell_single.shape) != [3, 3]:
                    raise ValueError(
                        f"each `cell` must be a (3 x 3) tensor, got at least one "
                        f"tensor with shape {tuple(cell_single.shape)}"
                    )

                if cell_single.dtype != self._dtype:
                    raise ValueError(
                        f"each `cell` must have the same type {self._dtype} as "
                        "positions, got at least one tensor of type "
                        f"{cell_single.dtype}"
                    )

                if cell_single.device != self._device:
                    raise ValueError(
                        f"each `cell` must be on the same device {self._device} as "
                        "positions, got at least one tensor with device "
                        f"{cell_single.device}"
                    )

            # check shape, dtype and device of charges
            if charges_single.dim() != 2:
                raise ValueError(
                    f"each `charges` needs to be a 2-dimensional tensor, got at least "
                    f"one tensor with {charges_single.dim()} dimension(s) and shape "
                    f"{tuple(charges_single.shape)}"
                )

            if list(charges_single.shape) != [num_atoms, charges_single.shape[1]]:
                raise ValueError(
                    f"each `charges` must be a (n_atoms x n_channels) tensor, with"
                    f"`n_atoms` being the same as the variable `positions`. Got at "
                    f"least one tensor with shape {tuple(charges_single.shape)} where "
                    f"positions contains {len(positions_single)} atoms"
                )

            if charges_single.dtype != self._dtype:
                raise ValueError(
                    f"each `charges` must have the same type {self._dtype} as "
                    f"positions, got at least one tensor of type {charges_single.dtype}"
                )

            if charges_single.device != self._device:
                raise ValueError(
                    f"each `charges` must be on the same device {self._device} as "
                    f"positions, got at least one tensor with device "
                    f"{charges_single.device}"
                )

            # check shape, dtype and device of neighbor_indices and neighbor_shifts
            if neighbor_indices_single is not None:
                if neighbor_shifts_single is None:
                    raise ValueError(
                        "Need to provide both `neighbor_indices` and `neighbor_shifts` "
                        "together."
                    )

                if neighbor_indices_single.shape[0] != 2:
                    raise ValueError(
                        "neighbor_indices is expected to have shape (2, num_neighbors)"
                        f", but got {tuple(neighbor_indices_single.shape)} for one "
                        "structure"
                    )

                if neighbor_shifts_single.shape[1] != 3:
                    raise ValueError(
                        "neighbor_shifts is expected to have shape (num_neighbors, 3)"
                        f", but got {tuple(neighbor_shifts_single.shape)} for one "
                        "structure"
                    )

                if neighbor_shifts_single.shape[0] != neighbor_indices_single.shape[1]:
                    raise ValueError(
                        f"`neighbor_indices` and `neighbor_shifts` need to have shapes "
                        f"(2, num_neighbors) and (num_neighbors, 3). For at least one"
                        f"structure, got {tuple(neighbor_indices_single.shape)} and "
                        f"{tuple(neighbor_shifts_single.shape)}, which is inconsistent"
                    )

                if neighbor_indices_single.device != self._device:
                    raise ValueError(
                        f"each `neighbor_indices` must be on the same device "
                        f"{self._device} as positions, got at least one tensor with "
                        f"device {neighbor_indices_single.device}"
                    )

                if neighbor_shifts_single.device != self._device:
                    raise ValueError(
                        f"each `neighbor_shifts` must be on the same device "
                        f"{self._device} as positions, got at least one tensor with "
                        f"device {neighbor_shifts_single.device}"
                    )

        return positions, cell, charges, neighbor_indices, neighbor_shifts

    def _compute_impl(
        self,
        positions: Union[List[torch.Tensor], torch.Tensor],
        charges: Union[Union[List[torch.Tensor], torch.Tensor]],
        cell: Union[None, List[torch.Tensor], torch.Tensor],
        neighbor_indices: Union[None, List[torch.Tensor], torch.Tensor],
        neighbor_shifts: Union[None, List[torch.Tensor], torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Check that all shapes, data types and devices are consistent
        # Furthermore, to handle the special case in which only the inputs for a single
        # structure are provided, turn inputs into a list to be consistent with the
        # more general case
        (
            positions,
            cell,
            charges,
            neighbor_indices,
            neighbor_shifts,
        ) = self._validate_compute_parameters(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_shifts=neighbor_shifts,
        )

        # compute and append into a list the features of each structure
        potentials = []
        for (
            positions_single,
            cell_single,
            charges_single,
            neighbor_indices_single,
            neighbor_shifts_single,
        ) in zip(positions, cell, charges, neighbor_indices, neighbor_shifts):
            # Compute the potentials
            potentials.append(
                self._compute_single_system(
                    positions=positions_single,
                    charges=charges_single,
                    cell=cell_single,
                    neighbor_indices=neighbor_indices_single,
                    neighbor_shifts=neighbor_shifts_single,
                )
            )

        # if only a single structure if provided as input, we directly return a single
        # tensor containing its features rather than a list of tensors
        if len(positions) == 1:
            return potentials[0]
        else:
            return potentials
