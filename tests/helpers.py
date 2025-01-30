"""Test utilities wrap common functions in the tests"""

import math
from pathlib import Path
from typing import Optional

import torch
from vesin import NeighborList

from torchpme._utils import _get_device, _get_dtype

SQRT3 = math.sqrt(3)

DIR_PATH = Path(__file__).parent
EXAMPLES = DIR_PATH / ".." / "examples"
COULOMB_TEST_FRAMES = EXAMPLES / "coulomb_test_frames.xyz"


def define_crystal(crystal_name="CsCl", dtype=None, device=None):
    device = _get_device(device)
    dtype = _get_dtype(dtype)

    # Define all relevant parameters (atom positions, charges, cell) of the reference
    # crystal structures for which the Madelung constants obtained from the Ewald sums
    # are compared with reference values.
    # see https://www.sciencedirect.com/science/article/pii/B9780128143698000078#s0015
    # More detailed values can be found in https://pubs.acs.org/doi/10.1021/ic2023852

    # Caesium-Chloride (CsCl) structure:
    # - Cubic unit cell
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    if crystal_name == "CsCl":
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
        charges = torch.tensor([-1.0, 1.0])
        cell = torch.eye(3)
        madelung_ref = 2.0353610945260
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a primitive unit cell
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_primitive":
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        charges = torch.tensor([1.0, -1.0])
        cell = torch.tensor([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]])  # fcc
        madelung_ref = 1.7475645946
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a cubic unit cell
    # - cubic unit cell
    # - 4 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_cubic":
        positions = torch.tensor(
            [
                [0.0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
        )
        charges = torch.tensor([+1.0, -1, -1, -1, +1, +1, +1, -1])
        cell = 2 * torch.eye(3)
        madelung_ref = 1.7475645946
        num_formula_units = 4

    # ZnS (zincblende) structure
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    # Remarks: we use a primitive unit cell which makes the lattice parameter of the
    # cubic cell equal to 2.
    elif crystal_name == "zincblende":
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
        charges = torch.tensor([1.0, -1])
        cell = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        madelung_ref = 2 * 1.6380550533 / SQRT3
        num_formula_units = 1

    # Wurtzite structure
    # - non-cubic unit cell (triclinic)
    # - 2 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "wurtzite":
        u = 3 / 8
        c = math.sqrt(1 / u)
        positions = torch.tensor(
            [
                [0.5, 0.5 / SQRT3, 0.0],
                [0.5, 0.5 / SQRT3, u * c],
                [0.5, -0.5 / SQRT3, 0.5 * c],
                [0.5, -0.5 / SQRT3, (0.5 + u) * c],
            ],
        )
        charges = torch.tensor([1.0, -1, 1, -1])
        cell = torch.tensor(
            [[0.5, -0.5 * SQRT3, 0], [0.5, 0.5 * SQRT3, 0], [0, 0, c]],
        )
        madelung_ref = 1.64132 / (u * c)
        num_formula_units = 2

    # Fluorite structure (e.g. CaF2 with Ca2+ and F-)
    # - non-cubic (fcc) unit cell
    # - 1 neutral molecule per unit cell
    # - Cation-Anion ratio of 1:2
    elif crystal_name == "fluorite":
        a = 5.463
        a = 1.0
        positions = a * torch.tensor(
            [[1 / 4, 1 / 4, 1 / 4], [3 / 4, 3 / 4, 3 / 4], [0, 0, 0]]
        )
        charges = torch.tensor([-1, -1, 2])
        cell = torch.tensor([[a, a, 0], [a, 0, a], [0, a, a]]) / 2.0
        madelung_ref = 11.6365752270768
        num_formula_units = 1

    # Copper(I)-Oxide structure (e.g. Cu2O with Cu+ and O2-)
    # - cubic unit cell
    # - 2 neutral molecules per unit cell
    # - Cation-Anion ratio of 2:1
    elif crystal_name == "cu2o":
        a = 1.0
        positions = a * torch.tensor(
            [
                [0, 0, 0],
                [1 / 2, 1 / 2, 1 / 2],
                [1 / 4, 1 / 4, 1 / 4],
                [1 / 4, 3 / 4, 3 / 4],
                [3 / 4, 1 / 4, 3 / 4],
                [3 / 4, 3 / 4, 1 / 4],
            ],
        )
        charges = torch.tensor([-2, -2, 1, 1, 1, 1])
        cell = a * torch.eye(3)
        madelung_ref = 10.2594570330750
        num_formula_units = 2

    # Wigner crystal in simple cubic structure.
    # Wigner crystals are equivalent to the Jellium or uniform electron gas models.
    # For the purpose of this test, we define them to be structures in which the ion
    # cores form a perfect lattice, while the electrons are uniformly distributed over
    # the cell. In some sources, the role of the positive and negative charges are
    # flipped. These structures are used to test the code for cases in which the total
    # charge of the particles is not zero.
    # Wigner crystal energies are taken from "Zero-Point Energy of an Electron Lattice"
    # by Rosemary A., Coldwell‐Horsfall and Alexei A. Maradudin (1960), eq. (A21).
    elif crystal_name == "wigner_sc":
        positions = torch.tensor([[0, 0, 0]])
        charges = torch.tensor([1.0])
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.7601188
        wigner_seiz_radius = (3 / (4 * torch.pi)) ** (1 / 3)
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 2.83730
        num_formula_units = 1

    # Wigner crystal in bcc structure (note: this is the most stable structure).
    # See description of "wigner_sc" for a general explanation on Wigner crystals.
    # Used to test the code for cases in which the unit cell has a nonzero net charge.
    elif crystal_name == "wigner_bcc":
        positions = torch.tensor([[0, 0, 0]])
        charges = torch.tensor([1.0])
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [1 / 2, 1 / 2, 1 / 2]])

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791860
        wigner_seiz_radius = (3 / (4 * torch.pi * 2)) ** (
            1 / 3
        )  # 2 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924
        num_formula_units = 1

    # Same as above, but now using a cubic unit cell rather than the primitive bcc cell
    elif crystal_name == "wigner_bcc_cubiccell":
        positions = torch.tensor([[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]])
        charges = torch.tensor([1.0, 1.0])
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791860
        wigner_seiz_radius = (3 / (4 * torch.pi * 2)) ** (
            1 / 3
        )  # 2 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924
        num_formula_units = 2

    # Wigner crystal in fcc structure
    # See description of "wigner_sc" for a general explanation on Wigner crystals.
    # Used to test the code for cases in which the unit cell has a nonzero net charge.
    elif crystal_name == "wigner_fcc":
        positions = torch.tensor([[0, 0, 0]])
        charges = torch.tensor([1.0])
        cell = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]]) / 2

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791753
        wigner_seiz_radius = (3 / (4 * torch.pi * 4)) ** (
            1 / 3
        )  # 4 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488
        num_formula_units = 1

    # Same as above, but now using a cubic unit cell rather than the primitive fcc cell
    elif crystal_name == "wigner_fcc_cubiccell":
        positions = 0.5 * torch.tensor([[0.0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]])
        charges = torch.tensor([1.0, 1, 1, 1])
        cell = torch.eye(3)

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791753
        wigner_seiz_radius = (3 / (4 * torch.pi * 4)) ** (
            1 / 3
        )  # 4 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488
        num_formula_units = 4

    else:
        raise ValueError(f"crystal_name = {crystal_name} is not supported!")

    charges = charges.reshape((-1, 1))

    return (
        positions.to(device=device, dtype=dtype),
        charges.to(device=device, dtype=dtype),
        cell.to(device=device, dtype=dtype),
        torch.tensor(madelung_ref, device=device, dtype=dtype),
        num_formula_units,
    )


def neighbor_list(
    positions: torch.tensor,
    periodic: bool = True,
    box: Optional[torch.tensor] = None,
    cutoff: Optional[float] = None,
    full_neighbor_list: bool = False,
    neighbor_shifts: bool = False,
) -> tuple[torch.tensor, torch.tensor]:
    if box is None:
        box = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)

    if cutoff is None:
        cell_dimensions = torch.linalg.norm(box, dim=1)
        cutoff_torch = torch.min(cell_dimensions) / 2 - 1e-6
        cutoff = cutoff_torch.item()

    nl = NeighborList(cutoff=cutoff, full_list=full_neighbor_list)
    neighbor_indices, d, S = nl.compute(
        points=positions.to(dtype=torch.float64, device="cpu"),
        box=box.to(dtype=torch.float64, device="cpu"),
        periodic=periodic,
        quantities="PdS",
    )

    neighbor_indices = torch.from_numpy(neighbor_indices.astype(int)).to(
        device=positions.device
    )
    d = torch.from_numpy(d).to(dtype=positions.dtype, device=positions.device)
    S = torch.from_numpy(S).to(dtype=positions.dtype, device=positions.device)

    if not neighbor_shifts:
        return neighbor_indices, d
    return neighbor_indices, S


def compute_distances(positions, neighbor_indices, cell=None, neighbor_shifts=None):
    """Compute pairwise distances."""
    atom_is = neighbor_indices[:, 0]
    atom_js = neighbor_indices[:, 1]

    pos_is = positions[atom_is]
    pos_js = positions[atom_js]

    distance_vectors = pos_js - pos_is

    if cell is not None and neighbor_shifts is not None:
        shifts = neighbor_shifts.type(cell.dtype)
        distance_vectors += shifts @ cell
    elif cell is not None and neighbor_shifts is None:
        raise ValueError("Provided `cell` but no `neighbor_shifts`.")
    elif cell is None and neighbor_shifts is not None:
        raise ValueError("Provided `neighbor_shifts` but no `cell`.")

    return torch.linalg.norm(distance_vectors, dim=1)
