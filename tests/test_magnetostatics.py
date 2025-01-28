import pytest
import torch

from torchpme.calculators import CalculatorDipole
from torchpme.potentials import PotentialDipole


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
        potential=PotentialDipole(dtype=torch.float64),
        full_neighbor_list=False,
        dtype=torch.float64,
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
    assert torch.isclose(
        result, expected_result, atol=1e-4
    ), f"Expected {expected_result}, but got {result}"


@pytest.mark.parametrize(
    "smearing, sr_potential",
    [
        (1e10, torch.tensor(-0.2656, dtype=torch.float64)),
        (1e-10, torch.tensor(0.0000, dtype=torch.float64)),
    ],
)
def test_magnetostatics_sr(smearing, sr_potential):
    calculator = CalculatorDipole(
        potential=PotentialDipole(smearing=smearing, dtype=torch.float64),
        full_neighbor_list=False,
        lr_wavelength=1.0,
        dtype=torch.float64,
    )
    pot = calculator._compute_rspace(
        dipoles=system.dipoles,
        neighbor_indices=system.neighbor_indices,
        neighbor_vectors=system.neighbor_vectors,
    )
    result = torch.einsum("ij,ij->i", pot, system.dipoles).sum()
    expected_result = sr_potential
    assert torch.isclose(
        result, expected_result, atol=1e-4
    ), f"Expected {expected_result}, but got {result}"


def test_magnetostatic_ewald():
    alpha = 1.0
    smearing = (1 / (2 * alpha**2)) ** 0.5
    calculator = CalculatorDipole(
        potential=PotentialDipole(smearing=smearing, dtype=torch.float64),
        full_neighbor_list=False,
        lr_wavelength=0.1,
        dtype=torch.float64,
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
    assert torch.isclose(
        result, expected_result, atol=1e-4
    ), f"Expected {expected_result}, but got {result}"
