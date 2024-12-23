import torch
from torch import profiler

from ..potentials import PotentialDipole


class CalculatorDipole(torch.nn.Module):
    """TODO: Add docstring"""

    def __init__(
        self,
        potential: PotentialDipole,
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
    ):
        super().__init__()
        # TorchScript requires to initialize all attributes in __init__
        self._device = torch.device("cpu")
        self._dtype = torch.float32

        self.potential = potential

        self.full_neighbor_list = full_neighbor_list

        self.prefactor = prefactor

    def _compute_rspace(
        self,
        dipoles: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """TODO: Add docstring"""
        # Compute the pair potential terms V(r_ij) for each pair of atoms (i,j)
        # contained in the neighbor list
        with profiler.record_function("compute bare potential"):
            if self.potential.smearing is None:
                potentials_bare_scalar, potentials_bare_tensor = (
                    self.potential.from_dist(neighbor_vectors)
                )
            else:
                raise NotImplementedError(
                    "TODO: Implement smearing for `compute_rspace`"
                )

        # Multiply the bare potential terms V(r_ij) with the corresponding dipoles
        # of ``atom j'' to obtain q_j*V(r_ij). Since each atom j can be a neighbor of
        # multiple atom i's, we need to access those from neighbor_indices
        atom_is = neighbor_indices[:, 0]
        atom_js = neighbor_indices[:, 1]
        with profiler.record_function("compute real potential"):
            contributions_is = dipoles[atom_js] * potentials_bare_scalar - torch.einsum(
                "ij,ijk->ik", dipoles[atom_js], potentials_bare_tensor
            )

        # For each atom i, add up all contributions of the form q_j*V(r_ij) for j
        # ranging over all of its neighbors.
        with profiler.record_function("assign potential"):
            potential = torch.zeros_like(dipoles)
            potential.index_add_(0, atom_is, contributions_is)
            # If we are using a half neighbor list, we need to add the contributions
            # from the "inverse" pairs (j, i) to the atoms i
            if not self.full_neighbor_list:
                contributions_js = dipoles[
                    atom_is
                ] * potentials_bare_scalar - torch.einsum(
                    "ij,ijk->ik", dipoles[atom_is], potentials_bare_tensor
                )
                potential.index_add_(0, atom_js, contributions_js)

        # Compensate for double counting of pairs (i,j) and (j,i)
        return potential / 2

    def _compute_kspace(
        self,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"`compute_kspace` not implemented for {self.__class__.__name__}"
        )

    def forward(
        self,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_vectors: torch.Tensor,
    ):
        r"""TODO: Add docstring"""
        # self._validate_compute_parameters(
        #     charges=charges,
        #     cell=cell,
        #     positions=positions,
        #     neighbor_indices=neighbor_indices,
        #     neighbor_distances=neighbor_distances,
        #     smearing=self.potential.smearing,
        # )

        # Compute short-range (SR) part using a real space sum
        potential_sr = self._compute_rspace(
            dipoles=dipoles,
            neighbor_indices=neighbor_indices,
            neighbor_vectors=neighbor_vectors,
        )

        if self.potential.smearing is None:
            return self.prefactor * potential_sr
        return None
        # Compute long-range (LR) part using a Fourier / reciprocal space sum
        # potential_lr = self._compute_kspace(
        #     charges=charges,
        #     cell=cell,
        #     positions=positions,
        # )

        # return self.prefactor * (potential_sr + potential_lr)

    # @staticmethod
    # def _validate_compute_parameters(
    #     dipoles: torch.Tensor,
    #     cell: torch.Tensor,
    #     positions: torch.Tensor,
    #     neighbor_indices: torch.Tensor,
    #     neighbor_vectors: torch.Tensor,
    #     smearing: Optional[float],
    # ) -> None:
    #     device = positions.device
    #     dtype = positions.dtype

    #     # check shape, dtype and device of positions
    #     num_atoms = len(positions)
    #     if list(positions.shape) != [len(positions), 3]:
    #         raise ValueError(
    #             "`positions` must be a tensor with shape [n_atoms, 3], got tensor "
    #             f"with shape {list(positions.shape)}"
    #         )

    #     # check shape, dtype and device of cell
    #     if list(cell.shape) != [3, 3]:
    #         raise ValueError(
    #             "`cell` must be a tensor with shape [3, 3], got tensor with shape "
    #             f"{list(cell.shape)}"
    #         )

    #     if cell.dtype != dtype:
    #         raise ValueError(
    #             f"type of `cell` ({cell.dtype}) must be same as `positions` ({dtype})"
    #         )

    #     if cell.device != device:
    #         raise ValueError(
    #             f"device of `cell` ({cell.device}) must be same as `positions` "
    #             f"({device})"
    #         )

    #     if smearing is not None and torch.equal(
    #         cell.det(), torch.tensor(0.0, dtype=cell.dtype, device=cell.device)
    #     ):
    #         raise ValueError(
    #             "provided `cell` has a determinant of 0 and therefore is not valid for "
    #             "periodic calculation"
    #         )

    #     # check shape, dtype & device of `charges`
    #     if charges.dim() != 2:
    #         raise ValueError(
    #             "`charges` must be a 2-dimensional tensor, got "
    #             f"tensor with {charges.dim()} dimension(s) and shape "
    #             f"{list(charges.shape)}"
    #         )

    #     if list(charges.shape) != [num_atoms, charges.shape[1]]:
    #         raise ValueError(
    #             "`charges` must be a tensor with shape [n_atoms, n_channels], with "
    #             "`n_atoms` being the same as the variable `positions`. Got tensor with "
    #             f"shape {list(charges.shape)} where positions contains "
    #             f"{len(positions)} atoms"
    #         )

    #     if charges.dtype != dtype:
    #         raise ValueError(
    #             f"type of `charges` ({charges.dtype}) must be same as `positions` "
    #             f"({dtype})"
    #         )

    #     if charges.device != device:
    #         raise ValueError(
    #             f"device of `charges` ({charges.device}) must be same as `positions` "
    #             f"({device})"
    #         )

    #     # check shape, dtype & device of `neighbor_indices` and `neighbor_distances`
    #     if neighbor_indices.shape[1] != 2:
    #         raise ValueError(
    #             "neighbor_indices is expected to have shape [num_neighbors, 2]"
    #             f", but got {list(neighbor_indices.shape)} for one "
    #             "structure"
    #         )

    #     if neighbor_indices.device != device:
    #         raise ValueError(
    #             f"device of `neighbor_indices` ({neighbor_indices.device}) must be "
    #             f"same as `positions` ({device})"
    #         )

    #     if neighbor_distances.shape != neighbor_indices[:, 0].shape:
    #         raise ValueError(
    #             "`neighbor_indices` and `neighbor_distances` need to have shapes "
    #             "[num_neighbors, 2] and [num_neighbors], but got "
    #             f"{list(neighbor_indices.shape)} and {list(neighbor_distances.shape)}"
    #         )

    #     if neighbor_distances.device != device:
    #         raise ValueError(
    #             f"device of `neighbor_distances` ({neighbor_distances.device}) must be "
    #             f"same as `positions` ({device})"
    #         )
