import pytest
import torch
from ase.io import read
from helpers import DIPOLES_TEST_FRAMES, neighbor_list

from torchpme.calculators import CalculatorDipole
from torchpme.potentials import PotentialDipole
from torchpme.prefactors import eV_A


def compute_distance_vectors(
    positions, neighbor_indices, cell=None, neighbor_shifts=None
):
    """Compute pairwise distance vectors."""
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

    return distance_vectors


class System:
    def __init__(self):
        self.cell = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64
        )
        self.dipoles = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float64
        )
        self.positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float64
        )
        self.neighbor_indices = torch.tensor(
            [[0, 1], [1, 2], [0, 2]], dtype=torch.int64
        )
        self.neighbor_vectors = torch.tensor(
            [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float64
        )


system = System()


def test_magnetostatics_direct():
    calculator = CalculatorDipole(
        potential=PotentialDipole(),
        full_neighbor_list=False,
    )
    pot = calculator(
        dipoles=system.dipoles,
        cell=system.cell,
        positions=system.positions,
        neighbor_indices=system.neighbor_indices,
        neighbor_vectors=system.neighbor_vectors,
    )
    result = torch.einsum("ij,ij->i", pot, system.dipoles).sum()
    expected_result = torch.tensor(-0.2656, dtype=torch.float64)
    assert torch.isclose(result, expected_result, atol=1e-4), (
        f"Expected {expected_result}, but got {result}"
    )


@pytest.mark.parametrize(
    ("smearing", "sr_potential"),
    [
        (1e10, torch.tensor(-0.2656, dtype=torch.float64)),
        (1e-10, torch.tensor(0.0000, dtype=torch.float64)),
    ],
)
def test_magnetostatics_sr(smearing, sr_potential):
    calculator = CalculatorDipole(
        potential=PotentialDipole(smearing=smearing),
        full_neighbor_list=False,
        lr_wavelength=1.0,
    )
    pot = calculator._compute_rspace(
        dipoles=system.dipoles,
        neighbor_indices=system.neighbor_indices,
        neighbor_vectors=system.neighbor_vectors,
    )
    result = torch.einsum("ij,ij->i", pot, system.dipoles).sum()
    expected_result = sr_potential
    assert torch.isclose(result, expected_result, atol=1e-4), (
        f"Expected {expected_result}, but got {result}"
    )


def test_magnetostatic_ewald():
    alpha = 1.0
    smearing = (1 / (2 * alpha**2)) ** 0.5
    calculator = CalculatorDipole(
        potential=PotentialDipole(smearing=smearing),
        full_neighbor_list=False,
        lr_wavelength=0.1,
    )
    pot = calculator(
        dipoles=system.dipoles,
        cell=system.cell,
        positions=system.positions,
        neighbor_indices=system.neighbor_indices,
        neighbor_vectors=system.neighbor_vectors,
    )
    result = torch.einsum("ij,ij->i", pot, system.dipoles).sum()
    # result is calculated using espressomd DipolarP3M with the same parameters and mesh
    # size 64
    expected_result = torch.tensor(-0.30848574939287954, dtype=torch.float64)
    assert torch.isclose(result, expected_result, atol=1e-4), (
        f"Expected {expected_result}, but got {result}"
    )


frames = read(DIPOLES_TEST_FRAMES, ":3")
cutoffs = [3.9986718930, 4.0000000000, 4.7363281250]
alphas = [0.8819831493, 0.8956299559, 0.7215211182]
energies = [frame.get_potential_energy() for frame in frames]
forces = [frame.get_forces() for frame in frames]


@pytest.mark.parametrize(
    ("frame", "cutoff", "alpha", "energy", "force"),
    zip(frames, cutoffs, alphas, energies, forces),
)
def test_magnetostatic_ewald_crystal(frame, cutoff, alpha, energy, force):
    smearing = (1 / (2 * alpha**2)) ** 0.5
    calc = CalculatorDipole(
        potential=PotentialDipole(smearing=smearing),
        full_neighbor_list=False,
        lr_wavelength=0.1,
        prefactor=eV_A,
    )
    positions = torch.tensor(frame.get_positions(), dtype=torch.float64)
    dipoles = torch.tensor(frame.get_array("dipoles"), dtype=torch.float64)
    cell = torch.tensor(frame.get_cell().array, dtype=torch.float64)
    neighbor_indices, neighbor_shifts = neighbor_list(
        positions=positions,
        periodic=True,
        box=cell,
        cutoff=cutoff,
        full_neighbor_list=False,
        neighbor_shifts=True,
    )
    positions.requires_grad = True
    neighbor_distance_vectors = compute_distance_vectors(
        positions=positions,
        neighbor_indices=neighbor_indices,
        cell=cell,
        neighbor_shifts=neighbor_shifts,
    )
    pot = calc(
        dipoles=dipoles,
        cell=cell,
        positions=positions,
        neighbor_indices=neighbor_indices,
        neighbor_vectors=neighbor_distance_vectors,
    )

    result = torch.einsum("ij,ij->", pot, dipoles)
    expected_result = torch.tensor(energy, dtype=torch.float64)
    assert torch.isclose(result, expected_result, atol=1e-4), (
        f"Expected {expected_result}, but got {result}"
    )

    forces = -torch.autograd.grad(result, positions)[0]
    expected_forces = torch.tensor(force, dtype=torch.float64)
    assert torch.allclose(forces, expected_forces, atol=1e-4), (
        f"Expected {expected_forces}, but got {forces}"
    )
