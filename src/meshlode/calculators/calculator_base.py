from typing import List, Optional, Tuple, Union

import torch

from meshlode.lib import InversePowerLawPotential


@torch.jit.script
def _1d_tolist(x: torch.Tensor) -> List[int]:
    """Auxilary function to convert 1d torch tensor to list of integers."""
    result: List[int] = []
    for i in x:
        result.append(i.item())
    return result


@torch.jit.script
def _is_subset(subset_candidate: List[int], superset: List[int]) -> bool:
    """Checks whether all elements of `subset_candidate` are part of `superset`."""
    for element in subset_candidate:
        if element not in superset:
            return False
    return True


class CalculatorBase(torch.nn.Module):
    """
    Base calculator

    :param all_types: Optional global list of all atomic types that should be considered
        for the computation. This option might be useful when running the calculation on
        subset of a whole dataset and it required to keep the shape of the output
        consistent. If this is not set the possible atomic types will be determined when
        calling the :meth:`compute()`.
    """

    name = "CalculatorBase"

    def __init__(
        self,
        all_types: Optional[List[int]] = None,
        exponent: float = 1.0,
    ):
        super().__init__()

        if all_types is None:
            self.all_types = None
        else:
            self.all_types = _1d_tolist(torch.unique(torch.tensor(all_types)))

        self.exponent = exponent
        self.potential = InversePowerLawPotential(exponent=exponent)

    def _get_requested_types(self, types: List[torch.Tensor]) -> List[int]:
        """Extract a list of all unique and present types from the list of types."""
        all_types = torch.hstack(types)
        types_requested = _1d_tolist(torch.unique(all_types))

        if self.all_types is not None:
            if not _is_subset(types_requested, self.all_types):
                raise ValueError(
                    f"Global list of types {self.all_types} does not contain all "
                    f"types for the provided systems {types_requested}."
                )
            return self.all_types
        else:
            return types_requested

    def _one_hot_charges(
        self,
        types: torch.Tensor,
        requested_types: List[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        n_types = len(requested_types)
        one_hot_charges = torch.zeros((len(types), n_types), dtype=dtype, device=device)

        for i_type, atomic_type in enumerate(requested_types):
            one_hot_charges[types == atomic_type, i_type] = 1.0

        return one_hot_charges

    def _validate_compute_parameters(
        self,
        types: Union[List[torch.Tensor], torch.Tensor],
        positions: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[None, List[torch.Tensor], torch.Tensor],
        charges: Union[None, List[torch.Tensor], torch.Tensor],
        neighbor_indices: Union[None, List[torch.Tensor], torch.Tensor],
        neighbor_shifts: Union[None, List[torch.Tensor], torch.Tensor],
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Union[List[None], List[torch.Tensor]],
        List[torch.Tensor],
        Union[List[None], List[torch.Tensor]],
        Union[List[None], List[torch.Tensor]],
    ]:
        # validate types and positions
        if not isinstance(types, list):
            types = [types]
        if not isinstance(positions, list):
            positions = [positions]

        if len(types) != len(positions):
            raise ValueError(
                f"Got inconsistent lengths of types ({len(types)}) "
                f"positions ({len(positions)})"
            )

        if cell is None:
            cell = len(types) * [None]
        elif not isinstance(cell, list):
            cell = [cell]

        if len(types) != len(cell):
            raise ValueError(
                f"Got inconsistent lengths of types ({len(types)}) and "
                f"cell ({len(cell)})"
            )

        if neighbor_indices is None:
            neighbor_indices = len(types) * [None]
        elif not isinstance(neighbor_indices, list):
            neighbor_indices = [neighbor_indices]

        if len(types) != len(neighbor_indices):
            raise ValueError(
                f"Got inconsistent lengths of types ({len(types)}) and "
                f"neighbor_indices ({len(neighbor_indices)})"
            )

        if neighbor_shifts is None:
            neighbor_shifts = len(types) * [None]
        elif not isinstance(neighbor_shifts, list):
            neighbor_shifts = [neighbor_shifts]

        if len(types) != len(neighbor_shifts):
            raise ValueError(
                f"Got inconsistent lengths of types ({len(types)}) and "
                f"neighbor_indices ({len(neighbor_shifts)})"
            )

        # Check that all inputs are consistent. We don't require and test that all
        # dtypes and devices are consistent if a list of inputs. Each single "frame" is
        # processed independently.
        for (
            types_single,
            positions_single,
            cell_single,
            neighbor_indices_single,
            neighbor_shifts_single,
        ) in zip(types, positions, cell, neighbor_indices, neighbor_shifts):
            if len(types_single.shape) != 1:
                raise ValueError(
                    "each `types` must be a 1 dimensional tensor, got at least "
                    f"one tensor with {len(types_single.shape)} dimensions"
                )

            if positions_single.shape != (len(types_single), 3):
                raise ValueError(
                    "each `positions` must be a (n_types x 3) tensor, got at least "
                    f"one tensor with shape {list(positions_single.shape)}"
                )

            if types_single.device != positions_single.device:
                raise ValueError(
                    f"Inconsistent devices of types ({types_single.device}) and "
                    f"positions ({positions_single.device})"
                )

            if cell_single is not None:
                if cell_single.shape != (3, 3):
                    raise ValueError(
                        "each `cell` must be a (3 x 3) tensor, got at least "
                        f"one tensor with shape {list(cell_single.shape)}"
                    )

                if cell_single.dtype != positions_single.dtype:
                    raise ValueError(
                        "`cell` must be have the same dtype as `positions`, got "
                        f"{cell_single.dtype} and {positions_single.dtype}"
                    )

                if types_single.device != cell_single.device:
                    raise ValueError(
                        f"Inconsistent devices of types ({types_single.device}) and "
                        f"cell ({cell_single.device})"
                    )

            if neighbor_indices_single is not None:
                # TODO test dtype

                if neighbor_indices_single.shape != (2, len(types_single)):
                    raise ValueError(
                        "Expected shape of neighbor_indices is "
                        f"{2, len(types_single)}, but got "
                        f"{list(neighbor_indices_single.shape)}"
                    )

                if types_single.device != neighbor_indices_single.device:
                    raise ValueError(
                        f"Inconsistent devices of types ({types_single.device}) and "
                        f"neighbor_indices ({neighbor_indices_single.device})"
                    )

            if neighbor_shifts_single is not None:
                # TODO test dtype

                if neighbor_shifts_single.shape != (3, len(types_single)):
                    raise ValueError(
                        "Expected shape of neighbor_shifts is "
                        f"{3, len(types_single)}, but got "
                        f"{list(neighbor_shifts_single.shape)}"
                    )

                if types_single.device != neighbor_shifts_single.device:
                    raise ValueError(
                        f"Inconsistent devices of types ({types_single.device}) and "
                        f"neighbor_shifts_single ({neighbor_shifts_single.device})"
                    )

        # If charges are not provided, we assume that all types are treated separately
        if charges is None:
            charges = []
            for types_single, positions_single in zip(types, positions):
                # One-hot encoding of charge information
                charges_single = self._one_hot_charges(
                    types=types_single,
                    requested_types=self._get_requested_types(types),
                    dtype=positions_single.dtype,
                    device=positions_single.device,
                )
                charges.append(charges_single)

        # If charges are provided, we need to make sure that they are consistent with
        # the provided types
        else:
            if not isinstance(charges, list):
                charges = [charges]
            if len(charges) != len(types):
                raise ValueError(
                    "The number of `types` and `charges` tensors must be the same, "
                    f"got {len(types)} and {len(charges)}."
                )
            for charges_single, types_single in zip(charges, types):
                if charges_single.shape[0] != len(types_single):
                    raise ValueError(
                        "The first dimension of `charges` must be the same as the "
                        f"length of `types`, got {charges_single.shape[0]} and "
                        f"{len(types_single)}."
                    )
            if charges[0].dtype != positions[0].dtype:
                raise ValueError(
                    "`charges` must be have the same dtype as `positions`, got "
                    f"{charges[0].dtype} and {positions[0].dtype}."
                )
            if charges[0].device != positions[0].device:
                raise ValueError(
                    "`charges` must be on the same device as `positions`, got "
                    f"{charges[0].device} and {positions[0].device}."
                )

        return types, positions, cell, charges, neighbor_indices, neighbor_shifts

    def _compute_impl(
        self,
        types: Union[List[torch.Tensor], torch.Tensor],
        positions: Union[List[torch.Tensor], torch.Tensor],
        cell: Union[List[torch.Tensor], torch.Tensor] = None,
        charges: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        neighbor_indices: Union[List[torch.Tensor], torch.Tensor] = None,
        neighbor_shifts: Union[List[torch.Tensor], torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        types, positions, cell, charges, neighbor_indices, neighbor_shifts = (
            self._validate_compute_parameters(
                types, positions, cell, charges, neighbor_indices, neighbor_shifts
            )
        )
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

        if len(types) == 1:
            return potentials[0]
        else:
            return potentials

    def _compute_single_system(
        self,
        positions: torch.Tensor,
        cell: Union[None, torch.Tensor],
        charges: torch.Tensor,
        neighbor_indices: Union[None, torch.Tensor],
        neighbor_shifts: Union[None, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError("only implemented in child classes")
